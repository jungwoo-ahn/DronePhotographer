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

    def forward(self, hidden_states, timestep, encoder_hidden_states, condition_mask=None, padding_mask=None, return_dict=True):
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


def test_action_scale_is_a_persisted_buffer():
    from src.policy.common.action_repr import ACTION_SCALE
    backbone = _DiffusersStyleMockBackbone()
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4)
    # registered as a buffer → present in state_dict → travels with the checkpoint
    assert "action_scale" in dict(policy.named_buffers())
    assert "action_scale" in policy.state_dict()
    torch.testing.assert_close(policy.action_scale.cpu().numpy(), ACTION_SCALE, atol=1e-6, rtol=0)


def test_custom_action_scale_overrides_default():
    backbone = _DiffusersStyleMockBackbone()
    custom = [1.0, 2.0, 3.0, 4.0, 5.0]
    policy = CosmosWorldActionPolicy(backbone, chunk_size=1, action_scale=custom)
    torch.testing.assert_close(policy.action_scale.cpu().tolist(), custom)


def test_sample_denormalize_scales_by_buffer():
    backbone = _DiffusersStyleMockBackbone()
    custom = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
    policy = CosmosWorldActionPolicy(backbone, chunk_size=4, action_scale=custom).eval()
    img = _make_image_latent()
    goal = torch.randn(2, 8)
    # Seed identically so both runs share the same sampling noise → only the
    # denormalize flag differs.
    torch.manual_seed(0)
    raw = policy.sample(img, goal, n_steps=2, denormalize=False).pred_action_chunk
    torch.manual_seed(0)
    denorm = policy.sample(img, goal, n_steps=2, denormalize=True).pred_action_chunk
    torch.testing.assert_close(denorm, raw * 10.0, atol=1e-4, rtol=1e-4)


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


# --- AdaLN-Zero goal conditioning ---------------------------------------------

class _MockTimeEmbed(nn.Module):
    """Mimics CosmosEmbedding: forward(hidden, timestep) -> (temb, embedded_timestep),
    with a `t_embedder.linear_2` so the model can auto-detect temb_dim."""

    def __init__(self, temb_dim: int):
        super().__init__()
        self.t_embedder = nn.Module()
        self.t_embedder.linear_2 = nn.Linear(1, temb_dim, bias=False)  # out_features = temb_dim
        self.temb_dim = temb_dim

    def forward(self, hidden_states, timestep):
        temb = timestep.float().view(-1, 1).expand(-1, self.temb_dim).to(hidden_states.dtype)
        return temb, temb


class _AdaLNMockBackbone(nn.Module):
    """Diffusers-style backbone whose output depends on `temb` (so an AdaLN goal
    injection actually changes it). `time_embed` is a submodule the model hooks."""

    def __init__(self, latent_channels=16, model_dim=1024, temb_dim=48):
        super().__init__()
        self.conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.time_embed = _MockTimeEmbed(temb_dim)
        self.temb_to_c = nn.Linear(temb_dim, latent_channels)
        self.cond_proj = nn.Linear(model_dim, latent_channels)

    def forward(self, hidden_states, timestep, encoder_hidden_states, condition_mask=None, padding_mask=None, return_dict=True):
        b, c, t, h, w = hidden_states.shape
        temb, _ = self.time_embed(hidden_states, timestep.flatten())   # hook adds goal to temb here
        mod = self.temb_to_c(temb).view(b, t, c).permute(0, 2, 1)[:, :, :, None, None]
        x = self.conv(hidden_states) + mod
        if encoder_hidden_states.shape[1] > 0:   # anchor-only context can be empty in tests
            x = x + self.cond_proj(encoder_hidden_states.mean(dim=1))[:, :, None, None, None]
        return _DiffusersOutput(sample=x) if return_dict else (x,)


def test_adaln_mode_drops_prefix_tokens_and_builds_goal_adaln():
    p = CosmosWorldActionPolicy(_AdaLNMockBackbone(), goal_conditioning="adaln",
                                adaln_temb_dim=48, chunk_size=1, freeze_backbone=False)
    assert p.conditioner.n_tokens == 0 and p.goal_adaln is not None    # goal not in cross-attn
    p2 = CosmosWorldActionPolicy(_DiffusersStyleMockBackbone(), n_goal_tokens=4)
    assert p2.conditioner.n_tokens == 4 and p2.goal_adaln is None       # default cross_attn intact


def test_adaln_temb_dim_autodetected_from_backbone():
    p = CosmosWorldActionPolicy(_AdaLNMockBackbone(temb_dim=48), goal_conditioning="adaln",
                                chunk_size=1)   # no adaln_temb_dim -> detect
    assert p.goal_adaln.proj[-1].out_features == 48


def test_adaln_goal_has_no_effect_at_init_timestep_preserved():
    p = CosmosWorldActionPolicy(_AdaLNMockBackbone(), goal_conditioning="adaln",
                                adaln_temb_dim=48, chunk_size=1, freeze_backbone=False).eval()
    img = _make_image_latent(t_img=2)
    g0, g1 = torch.zeros(2, 8), torch.randn(2, 8) * 5   # build goals BEFORE seeding the sampler noise
    torch.manual_seed(0); o1 = p.sample(img, g0, n_steps=2).pred_action_chunk
    torch.manual_seed(0); o2 = p.sample(img, g1, n_steps=2).pred_action_chunk
    torch.testing.assert_close(o1, o2)     # zero-init goal_adaln -> goal inert -> timestep path intact


def test_adaln_goal_engages_when_output_layer_activated():
    p = CosmosWorldActionPolicy(_AdaLNMockBackbone(), goal_conditioning="adaln",
                                adaln_temb_dim=48, chunk_size=1, freeze_backbone=False).eval()
    with torch.no_grad():
        p.goal_adaln.proj[-1].weight.normal_(0, 0.5)
    img = _make_image_latent(t_img=2)
    g0, g1 = torch.zeros(2, 8), torch.randn(2, 8) * 5   # build goals BEFORE seeding the sampler noise
    torch.manual_seed(0); o1 = p.sample(img, g0, n_steps=2).pred_action_chunk
    torch.manual_seed(0); o2 = p.sample(img, g1, n_steps=2).pred_action_chunk
    assert not torch.allclose(o1, o2)      # goal now modulates the prediction via AdaLN


def test_adaln_goal_adaln_gets_gradient_even_with_frozen_backbone():
    p = CosmosWorldActionPolicy(_AdaLNMockBackbone(), goal_conditioning="adaln",
                                adaln_temb_dim=48, chunk_size=1, freeze_backbone=True)
    img = _make_image_latent(t_img=2)
    p.compute_loss(img, torch.randn(2, 1, ACTION_DIM), torch.randn(2, 8)).total.backward()
    assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in p.goal_adaln.parameters())


def test_adaln_rejects_bad_conditioning_name():
    with pytest.raises(ValueError):
        CosmosWorldActionPolicy(_DiffusersStyleMockBackbone(), goal_conditioning="bogus")
