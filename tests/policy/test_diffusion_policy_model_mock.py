"""End-to-end DiffusionPolicy test with a mock DINOv2 backbone (no real weights)."""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from torch import nn

from src.policy.common.action_repr import ACTION_DIM
from src.policy.diffusion_policy.model import DiffusionPolicy, DPLossOutputs, DPOutputs


@dataclass
class _BackboneOut:
    pooler_output: torch.Tensor
    last_hidden_state: torch.Tensor


class _MockDINOv2(nn.Module):
    """Mimics a DINOv2 AutoModel: forward(pixel_values) -> obj with .pooler_output
    (B, D) and .last_hidden_state (B, L, D); exposes config.hidden_size."""

    def __init__(self, hidden=64, n_tokens=10):
        super().__init__()
        self.proj = nn.Linear(3, hidden)
        self.n_tokens = n_tokens
        self.config = type("cfg", (), {"hidden_size": hidden})()

    def forward(self, pixel_values=None, **kw):
        b = pixel_values.shape[0]
        feat = pixel_values.mean(dim=(2, 3))                  # (B, 3)
        h = self.proj(feat)[:, None, :].expand(b, self.n_tokens, -1).contiguous()
        return _BackboneOut(pooler_output=h.mean(dim=1), last_hidden_state=h)


def _policy(**kw):
    bb = _MockDINOv2(hidden=64)
    return DiffusionPolicy(bb, chunk_size=8, goal_embed_dim=32, down_dims=(64, 128, 256),
                           diffusion_step_embed_dim=64, num_train_timesteps=100, **kw)


def _inputs(b=2, h=16, w=16):
    return {"pixel_values": torch.randn(b, 3, h, w)}, torch.randn(b, 8)


def test_obs_dim_inferred_from_backbone():
    p = _policy()
    assert p.obs_dim == 64


def test_compute_loss_shape_and_backward_frozen():
    p = _policy(freeze_backbone=True)
    obs, goal = _inputs()
    action = torch.randn(2, 8, ACTION_DIM)
    out = p.compute_loss(obs, goal, action)
    assert isinstance(out, DPLossOutputs)
    assert out.total.dim() == 0
    out.total.backward()
    # frozen backbone gets NO grad; the trained head (denoiser + goal embed) does
    assert all(pp.grad is None or pp.grad.abs().sum() == 0 for pp in p.backbone.parameters())
    assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in p.denoiser.parameters())
    assert p.goal_embed[0].weight.grad is not None and p.goal_embed[0].weight.grad.abs().sum() > 0


def test_freeze_backbone_flag():
    p = _policy(freeze_backbone=True)
    assert all(not pp.requires_grad for pp in p.backbone.parameters())
    assert all(pp.requires_grad for pp in p.denoiser.parameters())
    pf = _policy(freeze_backbone=False)
    assert all(pp.requires_grad for pp in pf.backbone.parameters())


def test_sample_shape_and_denormalize():
    p = _policy().eval()
    obs, goal = _inputs()
    out = p.sample(obs, goal, n_steps=4)
    assert isinstance(out, DPOutputs)
    assert out.pred_action_chunk.shape == (2, 8, ACTION_DIM)
    torch.manual_seed(0); raw = p.sample(obs, goal, n_steps=4, denormalize=False).pred_action_chunk
    torch.manual_seed(0); den = p.sample(obs, goal, n_steps=4, denormalize=True).pred_action_chunk
    torch.testing.assert_close(den, raw * p.action_scale, atol=1e-4, rtol=1e-4)


def test_action_scale_is_persisted_buffer():
    from src.policy.common.action_repr import ACTION_SCALE
    p = _policy()
    assert "action_scale" in p.state_dict()
    torch.testing.assert_close(p.action_scale.cpu().numpy(), ACTION_SCALE, atol=1e-6, rtol=0)


def test_goal_changes_the_global_cond():
    p = _policy().eval()
    obs, _ = _inputs(b=1)
    c1 = p.global_cond(obs, torch.zeros(1, 8))
    c2 = p.global_cond(obs, torch.ones(1, 8))
    # the obs portion (first obs_dim) is identical; the goal-embed tail differs
    torch.testing.assert_close(c1[:, :p.obs_dim], c2[:, :p.obs_dim])
    assert not torch.allclose(c1[:, p.obs_dim:], c2[:, p.obs_dim:])


def test_fixed_timesteps_make_val_loss_deterministic():
    p = _policy(freeze_backbone=True).eval()
    obs, goal = _inputs()
    action = torch.randn(2, 8, ACTION_DIM)
    t = torch.full((2,), 50)
    torch.manual_seed(1); a = p.compute_loss(obs, goal, action, timesteps=t).total
    torch.manual_seed(1); b = p.compute_loss(obs, goal, action, timesteps=t).total
    torch.testing.assert_close(a, b)
