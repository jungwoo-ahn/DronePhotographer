"""Shape and behavior tests for src/policy/cosmos/conditioner.py.

GPU-free — pure torch.nn. Uses a synthetic anchor (no Qwen forward needed).

The conditioner emits ONLY real text tokens + K goal tokens (no padding). At init
the zero-init gate makes the goal tokens exactly zero, so the conditioned context
is the anchor text + K zero tokens regardless of the goal; a non-zero gate makes
the goal tokens depend on the goal.
"""

import pytest

torch = pytest.importorskip("torch")

from src.policy.cosmos.conditioner import (
    COSMOS_CROSSATTN_DIM,
    COSMOS_TEXT_MAX_LEN,
    DEFAULT_GOAL_TOKENS,
    ShotProfileVectorConditioner,
)


def _make_anchor(model_dim=COSMOS_CROSSATTN_DIM, max_len=COSMOS_TEXT_MAX_LEN, real_len=6):
    """Synthetic anchor: `max_len` rows, but only the first `real_len` are real text.
    The padding rows are deliberately non-zero so tests catch any code that uses them."""
    emb = torch.randn(max_len, model_dim)
    return emb, real_len


def _make_cond(**overrides):
    emb, real_len = _make_anchor(
        model_dim=overrides.get("model_dim", COSMOS_CROSSATTN_DIM),
        max_len=overrides.get("max_seq_len", COSMOS_TEXT_MAX_LEN),
        real_len=overrides.pop("anchor_real_len", 6),
    )
    cond = ShotProfileVectorConditioner(**overrides, anchor_embedding=emb, anchor_real_len=real_len)
    cond._test_anchor_full = emb            # stash the full anchor for assertions
    return cond


def test_output_shape_is_real_text_plus_goal_tokens():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    out = cond({"goal_vec": torch.randn(3, 8)})
    assert out.crossattn_emb.shape == (3, 6 + 4, COSMOS_CROSSATTN_DIM)   # no padding
    assert out.padding_mask.shape == (3, 6 + 4)
    assert out.raw_goal.shape == (3, 8)


def test_only_real_tokens_are_used_padding_is_ignored():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    out = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    # text region equals the anchor's first real_len rows (padding rows never appear)
    torch.testing.assert_close(out[:, :6], cond._test_anchor_full[:6].unsqueeze(0).expand(2, -1, -1))


def test_gate_initialized_to_zero():
    assert torch.all(_make_cond(goal_dim=8).gate == 0.0)


def test_goal_tokens_zero_at_init_regardless_of_goal():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6).eval()
    o1 = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    o2 = cond({"goal_vec": torch.randn(2, 8) * 100}).crossattn_emb
    # goal-token region is exactly zero at init, independent of the goal
    assert torch.all(o1[:, 6:] == 0.0)
    torch.testing.assert_close(o1, o2, atol=0, rtol=0)


def test_nonzero_gate_makes_goal_tokens_depend_on_goal():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    with torch.no_grad():
        cond.gate.fill_(1.0)
    o1 = cond({"goal_vec": torch.zeros(1, 8)}).crossattn_emb
    o2 = cond({"goal_vec": torch.randn(1, 8)}).crossattn_emb
    # text region unchanged; goal-token region changes with the goal
    torch.testing.assert_close(o1[:, :6], o2[:, :6])
    assert not torch.allclose(o1[:, 6:], o2[:, 6:])


def test_padding_mask_is_all_valid():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    mask = cond({"goal_vec": torch.randn(2, 8)}).padding_mask
    assert mask.shape == (2, 10) and torch.all(mask)   # no padding -> all True


def test_dropout_zeros_goal_tokens_in_train():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0, n_tokens=4, anchor_real_len=6).train()
    with torch.no_grad():
        cond.gate.fill_(10.0)
    out = cond({"goal_vec": torch.randn(8, 8)}).crossattn_emb
    assert torch.all(out[:, 6:] == 0.0)                # rate=1 -> every item dropped


def test_no_dropout_in_eval():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0, n_tokens=4, anchor_real_len=6).eval()
    with torch.no_grad():
        cond.gate.fill_(1.0)
    out = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    assert not torch.all(out[:, 6:] == 0.0)            # eval -> no dropout, goal active


def test_rejects_wrong_goal_dim():
    cond = _make_cond(goal_dim=8)
    with pytest.raises(ValueError):
        cond({"goal_vec": torch.randn(2, 5)})


def test_promotes_unbatched_input():
    out = _make_cond(goal_dim=4)({"goal_vec": torch.randn(4)})
    assert out.crossattn_emb.shape[0] == 1


def test_gradient_flows_to_goal_proj_and_gate():
    cond = _make_cond(goal_dim=4).train()
    with torch.no_grad():
        cond.gate.fill_(0.5)
    cond({"goal_vec": torch.randn(2, 4)}).crossattn_emb.pow(2).sum().backward()
    assert cond.goal_proj.weight.grad is not None and cond.goal_proj.weight.grad.abs().sum() > 0
    assert cond.gate.grad is not None and cond.gate.grad.abs().item() > 0


def test_rejects_real_len_longer_than_anchor():
    emb = torch.zeros(8, 32)
    with pytest.raises(ValueError):
        ShotProfileVectorConditioner(goal_dim=4, model_dim=32, n_tokens=2,
                                     anchor_embedding=emb, anchor_real_len=99)


def test_rejects_anchor_dim_mismatch():
    emb = torch.zeros(16, 32)
    with pytest.raises(ValueError):
        ShotProfileVectorConditioner(goal_dim=4, model_dim=64, n_tokens=2,
                                     anchor_embedding=emb, anchor_real_len=6)


def test_anchor_text_is_buffer_gate_and_proj_are_params():
    cond = _make_cond(goal_dim=8)
    params = {n for n, _ in cond.named_parameters()}
    buffers = {n for n, _ in cond.named_buffers()}
    assert "anchor_text" in buffers and "anchor_text" not in params
    assert "gate" in params and "goal_proj.weight" in params
