"""End-to-end VLAActionPolicy test with a mock Qwen-VL backbone (no 2B weights)."""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.policy.common.action_repr import ACTION_DIM
from src.policy.vla.model import VLAActionPolicy, VLALossOutputs, VLAOutputs


@dataclass
class _BackboneOut:
    last_hidden_state: torch.Tensor


class _MockQwenVL(nn.Module):
    """Mimics Qwen3VLModel: forward(**inputs) -> obj with .last_hidden_state (B, L, D).

    Derives a per-image token sequence from pixel_values so gradients flow back
    to the backbone (verifying full-finetune wiring), and exposes a `config` with
    `text_config.hidden_size` like the real model.
    """

    def __init__(self, hidden=64, n_tokens=12):
        super().__init__()
        self.proj = nn.Linear(3 * 4 * 4, hidden)
        self.n_tokens = n_tokens
        self.config = type("cfg", (), {"text_config": type("tc", (), {"hidden_size": hidden})()})()

    def forward(self, pixel_values=None, **kw):
        b = pixel_values.shape[0]
        # fake "patchify": L tokens, each from a random linear view of the image mean
        feat = pixel_values.mean(dim=(2, 3))                      # (B, 3)
        feat = feat[:, None, :].expand(b, self.n_tokens, 3)
        pad = torch.zeros(b, self.n_tokens, 3 * 4 * 4 - 3, device=pixel_values.device, dtype=pixel_values.dtype)
        tok = self.proj(torch.cat([feat, pad], dim=-1))
        return _BackboneOut(last_hidden_state=tok)


def _policy(**kw):
    bb = _MockQwenVL(hidden=64)
    return VLAActionPolicy(bb, chunk_size=8, expert_dim=32, expert_depth=2, expert_heads=4, **kw)


def _inputs(b=2, h=16, w=16):
    return {"pixel_values": torch.randn(b, 3, h, w)}, torch.randn(b, 8)


def test_ctx_dim_inferred_from_backbone():
    p = _policy()
    assert p.ctx_dim == 64


def test_compute_loss_shape_and_backward():
    p = _policy(freeze_backbone=False)
    vlm, goal = _inputs()
    action = torch.randn(2, 8, ACTION_DIM)
    out = p.compute_loss(vlm, goal, action)
    assert isinstance(out, VLALossOutputs)
    assert out.total.dim() == 0
    # break expert zero-init so the loss has gradient signal everywhere
    torch.nn.init.normal_(p.action_expert.out_proj.weight, std=0.02)
    out = p.compute_loss(vlm, goal, action)
    out.total.backward()
    assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in p.backbone.parameters()), "no VLM grad"
    assert p.goal_proj.weight.grad is not None and p.goal_proj.weight.grad.abs().sum() > 0


def test_freeze_backbone_disables_vlm_grads_only():
    p = _policy(freeze_backbone=True)
    assert all(not pp.requires_grad for pp in p.backbone.parameters())
    assert all(pp.requires_grad for pp in p.action_expert.parameters())
    assert p.goal_proj.weight.requires_grad


def test_sample_shape_and_denormalize():
    p = _policy().eval()
    vlm, goal = _inputs()
    out = p.sample(vlm, goal, n_steps=4)
    assert isinstance(out, VLAOutputs)
    assert out.pred_action_chunk.shape == (2, 8, ACTION_DIM)
    # denormalize scales by the action_scale buffer
    torch.manual_seed(0); raw = p.sample(vlm, goal, n_steps=4, denormalize=False).pred_action_chunk
    torch.manual_seed(0); den = p.sample(vlm, goal, n_steps=4, denormalize=True).pred_action_chunk
    torch.testing.assert_close(den, raw * p.action_scale, atol=1e-4, rtol=1e-4)


def test_action_scale_is_persisted_buffer():
    from src.policy.common.action_repr import ACTION_SCALE
    p = _policy()
    assert "action_scale" in p.state_dict()
    torch.testing.assert_close(p.action_scale.cpu().numpy(), ACTION_SCALE, atol=1e-6, rtol=0)


def test_goal_changes_the_context():
    p = _policy().eval()
    vlm, _ = _inputs(b=1)
    c1 = p.forward_context(vlm, torch.zeros(1, 8))
    c2 = p.forward_context(vlm, torch.ones(1, 8))
    # the goal-token tail (last n_goal_tokens positions) must differ
    n = p.n_goal_tokens
    assert not torch.allclose(c1[:, -n:], c2[:, -n:])
    # the VLM portion is identical (goal doesn't touch it)
    torch.testing.assert_close(c1[:, :-n], c2[:, :-n])
