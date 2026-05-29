"""End-to-end test for CosmosWorldActionPolicy with the latent-frame design.

Uses a mock backbone — the actual Cosmos-Predict2.5-2B isn't loaded here.
"""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.policy.common.action_repr import ACTION_DIM
from src.policy.cosmos.action_latent import extract_action_chunk, extract_value
from src.policy.cosmos.model import (
    CosmosNativeAdapter,
    CosmosWorldActionPolicy,
    DiffusersStyleAdapter,
    PolicyOutputs,
)


@dataclass
class _DiffusersOutput:
    sample: torch.Tensor


class _DiffusersStyleMockBackbone(nn.Module):
    """Diffusers-style: kwargs (hidden_states, timestep, encoder_hidden_states)."""

    def __init__(self, latent_channels: int = 16, model_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.cond_proj = nn.Linear(model_dim, latent_channels)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
        x = self.conv(hidden_states)
        c_emb = self.cond_proj(encoder_hidden_states.mean(dim=1))
        x = x + c_emb[:, :, None, None, None]
        return _DiffusersOutput(sample=x) if return_dict else (x,)


class _CosmosNativeMockBackbone(nn.Module):
    def __init__(self, latent_channels: int = 16, model_dim: int = 1024):
        super().__init__()
        self.conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.cond_proj = nn.Linear(model_dim, latent_channels)

    def forward(self, *, x_B_C_T_H_W, timesteps_B_T, crossattn_emb, padding_mask=None, **_):
        c_emb = self.cond_proj(crossattn_emb.mean(dim=1))
        return self.conv(x_B_C_T_H_W) + c_emb[:, :, None, None, None]


def _make_image_latent(b=2, c=16, t_img=4, h=8, w=8):
    return torch.randn(b, c, t_img, h, w)


def test_build_training_latents_includes_action_and_value_frames():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4, use_value_latent=True)
    img = _make_image_latent()
    action = torch.randn(2, 4, ACTION_DIM)
    value = torch.tensor([0.5, -0.3])
    x0 = policy.build_training_latents(img, action, value)
    # T_total = T_img + 2 (action + value)
    assert x0.shape == (2, 16, 6, 8, 8)
    # Image latents preserved at the start
    torch.testing.assert_close(x0[:, :, :4], img)
    # Action chunk recoverable
    extracted = extract_action_chunk(x0[:, :, 4], chunk_size=4, action_dim=ACTION_DIM)
    torch.testing.assert_close(extracted, action, atol=1e-5, rtol=0)
    # Value recoverable
    extracted_v = extract_value(x0[:, :, 5])
    torch.testing.assert_close(extracted_v, value, atol=1e-5, rtol=0)


def test_build_training_latents_without_value():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, use_value_latent=False)
    img = _make_image_latent()
    action = torch.randn(2, 1, ACTION_DIM)
    x0 = policy.build_training_latents(img, action)
    # T_total = T_img + 1 (action only)
    assert x0.shape == (2, 16, 5, 8, 8)


def test_compute_loss_returns_per_component_breakdown():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, freeze_backbone=False, chunk_size=1)
    img = _make_image_latent()
    action = torch.randn(2, 1, ACTION_DIM)
    goal = torch.randn(2, 8)
    out = policy.compute_loss(img, action, goal)
    assert out.total.dim() == 0
    assert out.world.dim() == 0
    assert out.action.dim() == 0
    assert out.value is not None and out.value.dim() == 0
    out.total.backward()
    has_backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.transformer.parameters())
    assert has_backbone_grad


def test_compute_loss_respects_lambda_weights():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(
        backbone, freeze_backbone=False, chunk_size=1,
        lambda_world=2.0, lambda_action=3.0, lambda_value=0.5,
    )
    img = _make_image_latent()
    action = torch.randn(2, 1, ACTION_DIM)
    goal = torch.randn(2, 8)
    out = policy.compute_loss(img, action, goal)
    expected = 2.0 * out.world + 3.0 * out.action + 0.5 * out.value
    torch.testing.assert_close(out.total, expected, atol=1e-5, rtol=1e-5)


def test_compute_loss_without_value_omits_value_term():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, use_value_latent=False, freeze_backbone=False)
    img = _make_image_latent()
    action = torch.randn(2, 1, ACTION_DIM)
    goal = torch.randn(2, 8)
    out = policy.compute_loss(img, action, goal)
    assert out.value is None
    # Total = lambda_world * world + lambda_action * action
    expected = policy.lambda_world * out.world + policy.lambda_action * out.action
    torch.testing.assert_close(out.total, expected, atol=1e-5, rtol=1e-5)


def test_sample_produces_action_chunk_and_value():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4, use_value_latent=True).eval()
    img = _make_image_latent()
    goal = torch.randn(2, 8)
    out: PolicyOutputs = policy.sample(img, goal, n_steps=4)
    assert out.pred_action_chunk.shape == (2, 4, ACTION_DIM)
    assert out.pred_value.shape == (2,)
    assert out.pred_latents.shape == (2, 16, 6, 8, 8)


def test_sample_pins_image_latent_frames_each_step():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=1).eval()
    img = _make_image_latent()
    goal = torch.randn(2, 8)
    out = policy.sample(img, goal, n_steps=8)
    # The first T_img frames should equal the conditioning input — they're re-pinned each step
    torch.testing.assert_close(out.pred_latents[:, :, :4], img, atol=1e-5, rtol=0)


def test_freeze_backbone_disables_grads():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, freeze_backbone=True)
    for p in policy.transformer.parameters():
        assert not p.requires_grad
    for p in policy.conditioner.parameters():
        assert p.requires_grad


def test_cosmos_native_adapter_works():
    backbone = _CosmosNativeMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, adapter="cosmos_native", chunk_size=1, freeze_backbone=False)
    img = _make_image_latent()
    action = torch.randn(2, 1, ACTION_DIM)
    goal = torch.randn(2, 8)
    out = policy.compute_loss(img, action, goal)
    assert out.total.dim() == 0
    out.total.backward()


def test_make_adapter_unknown_style_raises():
    backbone = _DiffusersStyleMockBackbone()
    with pytest.raises(ValueError):
        CosmosWorldActionPolicy(backbone, adapter="nonexistent")


def test_action_latent_indices_offset_correctly():
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, use_value_latent=True)
    # action at T_img, value at T_img+1
    assert policy.action_latent_idx(t_img=4) == 4
    assert policy.value_latent_idx(t_img=4) == 5
    assert policy.num_extra_latent_frames() == 2

    no_value = CosmosWorldActionPolicy(backbone, use_value_latent=False)
    assert no_value.num_extra_latent_frames() == 1
    with pytest.raises(ValueError):
        no_value.value_latent_idx(t_img=4)
