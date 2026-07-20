"""Shape and behavior tests for src/policy/cosmos/conditioner.py.

GPU-free — pure torch.nn. Uses a synthetic anchor (no Qwen forward needed).

Conditional prefix tuning (magnitude-matched): the conditioner emits ONLY real text
tokens + K goal tokens. The goal tokens are RMSNorm'd to the anchor's per-token scale
(L2 ≈ sqrt(model_dim)) × a learnable gain (init 1), so they are goal-dependent and
audible from step 1 — the fix for the old zero-init prefix that sat ~1700× below the
anchor. `goal_proj` learns token direction; the gain learns loudness.
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


def test_goal_proj_not_zero_init_and_has_learnable_gain():
    cond = _make_cond(goal_dim=8)
    # goal_proj learns DIRECTION -> must be non-zero init (RMSNorm sets the magnitude).
    assert cond.goal_proj.weight.abs().sum() > 0.0
    assert not hasattr(cond, "gate")                   # the scalar gate is gone
    assert hasattr(cond, "goal_gain") and float(cond.goal_gain) == pytest.approx(1.0)


def test_goal_tokens_are_anchor_scale_and_goal_dependent_at_init():
    model_dim = COSMOS_CROSSATTN_DIM
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6, model_dim=model_dim).eval()
    o1 = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    o2 = cond({"goal_vec": torch.randn(2, 8) * 100}).crossattn_emb
    # Goal tokens depend on the goal from step 1 (no zero-init blackout).
    assert not torch.allclose(o1[:, 6:], o2[:, 6:])
    # And they sit at the anchor's per-token scale: L2 ≈ sqrt(model_dim) (gain init 1).
    goal_tok_l2 = o1[:, 6:].norm(dim=-1)               # (B, K)
    torch.testing.assert_close(
        goal_tok_l2, torch.full_like(goal_tok_l2, model_dim ** 0.5), rtol=1e-3, atol=1e-2)


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


def test_goal_proj_and_gain_get_gradient():
    """Both the direction (goal_proj) and the loudness (goal_gain) receive gradient
    from step 1 — the RMSNorm + gain path is fully differentiable, no deadlock."""
    cond = _make_cond(goal_dim=4, n_tokens=4, anchor_real_len=6).train()
    out = cond({"goal_vec": torch.randn(3, 4)})
    out.crossattn_emb.sum().backward()                  # linear loss
    assert cond.goal_proj.weight.grad is not None and cond.goal_proj.weight.grad.abs().sum() > 0
    assert cond.goal_gain.grad is not None and cond.goal_gain.grad.abs() > 0


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


# --- AdaLN-Zero goal conditioner (the alternative to cross-attention prefix) ---

def test_goal_adaln_is_zero_at_init_but_output_layer_gets_gradient():
    from src.policy.cosmos.conditioner import GoalAdaLNConditioner
    m = GoalAdaLNConditioner(goal_dim=8, temb_dim=48, hidden_dim=32)
    out = m(torch.randn(3, 8))
    assert out.shape == (3, 48)
    assert torch.all(out == 0.0)                       # zero-init -> timestep path untouched
    out.sum().backward()
    # AdaLN-Zero: the zero-init OUTPUT layer still gets gradient from step 1
    assert m.proj[-1].weight.grad is not None and m.proj[-1].weight.grad.abs().sum() > 0


def test_goal_adaln_depends_on_goal_once_output_layer_activated():
    from src.policy.cosmos.conditioner import GoalAdaLNConditioner
    m = GoalAdaLNConditioner(goal_dim=8, temb_dim=48)
    with torch.no_grad():
        m.proj[-1].weight.normal_(0, 0.1)
    assert not torch.allclose(m(torch.zeros(1, 8)), m(torch.randn(1, 8)))


def test_force_uncond_zeros_goal_tokens_even_in_eval():
    """The CFG null pass: force_uncond zeros every goal token, in eval mode too
    (where training dropout is inert)."""
    from src.policy.cosmos.conditioner import ShotProfileVectorConditioner
    c = ShotProfileVectorConditioner(goal_dim=8, model_dim=64, n_tokens=4).eval()
    g = torch.randn(2, 8)
    out = c({"goal_vec": g}, force_uncond=True)   # no anchor -> emb is just the goal tokens
    assert torch.allclose(out.crossattn_emb, torch.zeros_like(out.crossattn_emb))
    assert c({"goal_vec": g}).crossattn_emb.abs().sum() > 0


def test_conditioner_handles_32_goal_tokens():
    from src.policy.cosmos.conditioner import ShotProfileVectorConditioner
    c = ShotProfileVectorConditioner(goal_dim=8, model_dim=64, n_tokens=32)
    assert c({"goal_vec": torch.randn(3, 8)}).crossattn_emb.shape == (3, 32, 64)
