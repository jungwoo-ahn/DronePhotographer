"""Shape and behavior tests for src/policy/cosmos/conditioner.py.

GPU-free — pure torch.nn. Uses a synthetic anchor (no T5 forward needed).
"""

import pytest

torch = pytest.importorskip("torch")

from src.policy.cosmos.conditioner import (
    COSMOS_CROSSATTN_DIM,
    COSMOS_T5_MAX_LEN,
    DEFAULT_GOAL_TOKENS,
    ShotProfileVectorConditioner,
)


def _make_anchor(model_dim=COSMOS_CROSSATTN_DIM, max_len=COSMOS_T5_MAX_LEN, real_len=6):
    """Synthetic T5-like anchor for tests."""
    emb = torch.randn(max_len, model_dim)
    # Zero out the "padding" region so we can detect goal-token injection cleanly
    emb[real_len:] = 0.0
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:real_len] = True
    return emb, real_len, mask


def _make_cond(**overrides):
    """Helper: instantiate conditioner with a synthetic anchor."""
    emb, real_len, mask = _make_anchor(
        model_dim=overrides.get("model_dim", COSMOS_CROSSATTN_DIM),
        max_len=overrides.get("max_seq_len", COSMOS_T5_MAX_LEN),
        real_len=overrides.pop("anchor_real_len", 6),
    )
    return ShotProfileVectorConditioner(
        **overrides,
        anchor_embedding=emb,
        anchor_real_len=real_len,
        anchor_padding_mask=mask,
    )


def test_conditioner_output_shape():
    cond = _make_cond(goal_dim=8)
    goal = torch.randn(3, 8)
    out = cond({"goal_vec": goal})
    assert out.crossattn_emb.shape == (3, COSMOS_T5_MAX_LEN, COSMOS_CROSSATTN_DIM)
    assert out.padding_mask.shape == (3, COSMOS_T5_MAX_LEN)
    assert out.raw_goal.shape == (3, 8)


def test_gate_initialized_to_zero():
    cond = _make_cond(goal_dim=8)
    assert torch.all(cond.gate == 0.0)


def test_output_equals_anchor_at_init_regardless_of_goal():
    """The zero-init gate property: model starts as exactly the pretrained T5 path."""
    cond = _make_cond(goal_dim=8)
    cond.eval()
    g1 = torch.randn(2, 8)
    g2 = torch.randn(2, 8) * 100
    out1 = cond({"goal_vec": g1}).crossattn_emb
    out2 = cond({"goal_vec": g2}).crossattn_emb
    # Both should equal the broadcast anchor exactly
    expected = cond.anchor_emb.unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(out1, expected, atol=0, rtol=0)
    torch.testing.assert_close(out2, expected, atol=0, rtol=0)


def test_nonzero_gate_changes_output_at_goal_positions():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    with torch.no_grad():
        cond.gate.fill_(1.0)
    out = cond({"goal_vec": torch.randn(1, 8)}).crossattn_emb
    # Outside the goal-token range [6, 6+4), output equals anchor
    torch.testing.assert_close(out[0, :6], cond.anchor_emb[:6])
    torch.testing.assert_close(out[0, 10:], cond.anchor_emb[10:])
    # Inside the goal range, output differs from the (zero) anchor region
    assert not torch.allclose(out[0, 6:10], cond.anchor_emb[6:10])


def test_padding_mask_marks_goal_positions_valid():
    cond = _make_cond(goal_dim=8, n_tokens=4, anchor_real_len=6)
    out = cond({"goal_vec": torch.randn(2, 8)})
    mask = out.padding_mask
    # Anchor real-text positions are valid
    assert torch.all(mask[:, :6])
    # Goal positions are valid
    assert torch.all(mask[:, 6:10])
    # Beyond, padding stays masked
    assert torch.all(~mask[:, 10:])


def test_conditioner_rejects_wrong_goal_dim():
    cond = _make_cond(goal_dim=8)
    with pytest.raises(ValueError):
        cond({"goal_vec": torch.randn(2, 5)})


def test_conditioner_promotes_unbatched_input():
    cond = _make_cond(goal_dim=4)
    out = cond({"goal_vec": torch.randn(4)})
    assert out.crossattn_emb.shape[0] == 1


def test_dropout_zeros_gate_contribution_in_train():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0)
    cond.train()
    with torch.no_grad():
        cond.gate.fill_(10.0)
    out = cond({"goal_vec": torch.randn(8, 8)}).crossattn_emb
    expected = cond.anchor_emb.unsqueeze(0).expand(8, -1, -1)
    # rate=1.0 → every item drops gate effect → emb = anchor
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_no_dropout_in_eval_mode_even_with_high_rate():
    cond = _make_cond(goal_dim=8, dropout_rate=1.0)
    cond.eval()
    with torch.no_grad():
        cond.gate.fill_(1.0)
    out = cond({"goal_vec": torch.randn(2, 8)}).crossattn_emb
    expected = cond.anchor_emb.unsqueeze(0).expand(2, -1, -1)
    # In eval, no dropout → gate=1 changes the goal-position values
    assert not torch.allclose(out, expected)


def test_override_dropout_rate():
    cond = _make_cond(goal_dim=8, dropout_rate=0.0)
    cond.train()
    with torch.no_grad():
        cond.gate.fill_(1.0)
    out = cond({"goal_vec": torch.randn(8, 8)}, override_dropout_rate=1.0).crossattn_emb
    expected = cond.anchor_emb.unsqueeze(0).expand(8, -1, -1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_gradient_flows_to_goal_proj_and_gate():
    cond = _make_cond(goal_dim=4)
    cond.train()
    with torch.no_grad():
        cond.gate.fill_(0.5)  # so gradient through gate is non-zero
    goal = torch.randn(2, 4)
    out = cond({"goal_vec": goal})
    # Squared norm avoids LayerNorm zero-grad pathologies (and there's no LN here anyway)
    out.crossattn_emb.pow(2).sum().backward()
    assert cond.goal_proj.weight.grad is not None
    assert cond.goal_proj.weight.grad.abs().sum() > 0
    assert cond.gate.grad is not None
    assert cond.gate.grad.abs().item() > 0


def test_rejects_oversized_anchor_plus_tokens():
    emb = torch.zeros(16, 32)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[:14] = True
    with pytest.raises(ValueError):
        ShotProfileVectorConditioner(
            goal_dim=4, model_dim=32, max_seq_len=16, n_tokens=4,
            anchor_embedding=emb, anchor_real_len=14, anchor_padding_mask=mask,
        )


def test_rejects_anchor_shape_mismatch():
    emb = torch.zeros(16, 32)
    mask = torch.zeros(16, dtype=torch.bool)
    with pytest.raises(ValueError):
        ShotProfileVectorConditioner(
            goal_dim=4, model_dim=64, max_seq_len=16, n_tokens=2,
            anchor_embedding=emb, anchor_real_len=0, anchor_padding_mask=mask,
        )


def test_anchor_and_mask_are_buffers_not_parameters():
    cond = _make_cond(goal_dim=8)
    param_names = {n for n, _ in cond.named_parameters()}
    buffer_names = {n for n, _ in cond.named_buffers()}
    assert "anchor_emb" in buffer_names
    assert "anchor_mask" in buffer_names
    assert "anchor_emb" not in param_names
    assert "anchor_mask" not in param_names
    # And gate / proj are parameters
    assert "gate" in param_names
    assert "goal_proj.weight" in param_names
