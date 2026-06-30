"""Shape and behavior tests for src/policy/cosmos/conditioner.py.

GPU-free — pure torch.nn. Uses a synthetic anchor (no Qwen forward needed).

Conditional prefix tuning (no gate): the conditioner emits ONLY real text tokens +
K goal tokens. `goal_proj` is zero-initialized, so the prefix is exactly zero at
init (output = anchor text + K zero tokens, independent of the goal), yet `goal_proj`
still receives gradient from step 1 — no zero-init deadlock.
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
    emb = torch.randn(max_len, model_dim)
    return emb, real_len


def _make_cond(**overrides):
    emb, real_len = _make_anchor(
        model_dim=overrides.get("model_dim", COSMOS_CROSSATTN_DIM),
        max_len=overrides.get("max_seq_len", COSMOS_TEXT_MAX_LEN),
        real_len=overrides.pop("anchor_real_len", 6),
    )
    cond = ShotProfileVectorConditioner(**overrides, anchor_embedding=emb, anchor_real_len=real_len)
    cond._test_anchor_full = emb
    return cond


def _activate(cond, std=0.1):
    """Break the zero init so goal tokens become non-trivial (simulates trained proj)."""
    with torch.no_grad():
        cond.goal_proj.weight.normal_(0, std)
        cond.goal_proj.bias.normal_(0, std)


def test_output_shape_is_real_text_plus_goal_tokens():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    out = cond({"goal_vec": torch.randn(3, 8)})
    assert out.crossattn_emb.shape == (3, 6 + 4, COSMOS_CROSSATTN_DIM)   # no padding
    assert out.padding_mask.shape == (3, 6 + 4)
    assert out.raw_goal.shape == (3, 8)


def test_only_real_tokens_are_used_padding_is_ignored():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    out = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    torch.testing.assert_close(out[:, :6], cond._test_anchor_full[:6].unsqueeze(0).expand(2, -1, -1))


def test_goal_proj_zero_initialized():
    cond = _make_cond(goal_dim=8)
    assert torch.all(cond.goal_proj.weight == 0.0) and torch.all(cond.goal_proj.bias == 0.0)
    assert not hasattr(cond, "gate")   # the scalar gate is gone


def test_goal_tokens_zero_at_init_regardless_of_goal():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6).eval()
    o1 = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    o2 = cond({"goal_vec": torch.randn(2, 8) * 100}).crossattn_emb
    assert torch.all(o1[:, 6:] == 0.0)                 # zero-init proj -> zero prefix
    torch.testing.assert_close(o1, o2, atol=0, rtol=0)


def test_activated_proj_makes_goal_tokens_depend_on_goal():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    _activate(cond)
    o1 = cond({"goal_vec": torch.zeros(1, 8)}).crossattn_emb
    o2 = cond({"goal_vec": torch.randn(1, 8)}).crossattn_emb
    torch.testing.assert_close(o1[:, :6], o2[:, :6])    # text unchanged
    assert not torch.allclose(o1[:, 6:], o2[:, 6:])     # goal tokens depend on goal


def test_padding_mask_is_all_valid():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    mask = cond({"goal_vec": torch.randn(2, 8)}).padding_mask
    assert mask.shape == (2, 10) and torch.all(mask)


def test_goal_proj_gets_gradient_AT_ZERO_INIT():
    """The point of dropping the gate: gradient reaches goal_proj from step 1, even
    though the prefix is zero-initialized (a linear loss => d/dW = upstream · goal)."""
    cond = _make_cond(goal_dim=4, n_tokens=4, anchor_real_len=6).train()
    out = cond({"goal_vec": torch.randn(3, 4)})
    out.crossattn_emb.sum().backward()                  # linear loss
    assert cond.goal_proj.weight.grad is not None and cond.goal_proj.weight.grad.abs().sum() > 0
    assert cond.goal_proj.bias.grad is not None and cond.goal_proj.bias.grad.abs().sum() > 0


def test_dropout_zeros_goal_tokens_in_train():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0, n_tokens=4, anchor_real_len=6).train()
    _activate(cond)
    out = cond({"goal_vec": torch.randn(8, 8)}).crossattn_emb
    assert torch.all(out[:, 6:] == 0.0)                 # rate=1 -> every item dropped


def test_no_dropout_in_eval():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0, n_tokens=4, anchor_real_len=6).eval()
    _activate(cond)
    out = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    assert not torch.all(out[:, 6:] == 0.0)


def test_rejects_wrong_goal_dim():
    with pytest.raises(ValueError):
        _make_cond(goal_dim=8)({"goal_vec": torch.randn(2, 5)})


def test_promotes_unbatched_input():
    out = _make_cond(goal_dim=4)({"goal_vec": torch.randn(4)})
    assert out.crossattn_emb.shape[0] == 1


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


def test_anchor_text_buffer_goal_proj_param_no_gate():
    cond = _make_cond(goal_dim=8)
    params = {n for n, _ in cond.named_parameters()}
    buffers = {n for n, _ in cond.named_buffers()}
    assert "anchor_text" in buffers and "anchor_text" not in params
    assert "goal_proj.weight" in params
    assert "gate" not in params and "gate" not in buffers
