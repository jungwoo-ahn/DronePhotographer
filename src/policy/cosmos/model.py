"""CosmosWorldActionPolicy — Cosmos-Predict2.5 backbone with action+value as latent frames.

Issue #18 ("Action head: Use a similar action head, change the number of actions"):
we now follow cosmos-policy's pattern exactly — action and value live as extra
latent frames in the video sequence, predicted by the same flow-matching head
that predicts the image latents. There are no separate action / value MLPs.

Latent sequence layout (training):
  pos 0          .. T_img - 1   : image latents from the VAE
  pos T_img                      : action latent (filled with tiled action chunk)
  pos T_img + 1                  : value latent (filled with scalar return)

At training:
  - Encode images with the VAE → (B, 16, T_img, H_lat, W_lat).
  - Append two blank latent frames (zeros) for action / value → T_total = T_img + 2.
  - Inject the ground-truth action chunk + value via `action_latent.inject_*`.
  - Sample noise, interpolate, run the backbone, compute flow-matching MSE.

At inference:
  - Start from a noise tensor of shape (B, 16, T_total, H_lat, W_lat).
  - Optionally pin the image-conditioning frames (T<T_img) to the encoded state.
  - Run the backbone for `n_sampler_steps` (rectified-flow style for the
    prototype; we can swap in the Cosmos sampler later).
  - Extract the action chunk and value via `action_latent.extract_*`.

The whole design is parameterless beyond the backbone + conditioner — the
action / value heads are gone, replaced by latent-position indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch
from torch import nn

from src.policy.common.action_repr import ACTION_DIM, ACTION_SCALE
from src.policy.cosmos.action_latent import (
    extract_action_chunk,
    extract_value,
    inject_action_chunk,
    inject_value,
)
from src.policy.cosmos.conditioner import (
    COSMOS_CROSSATTN_DIM,
    ShotProfileCondition,
    ShotProfileVectorConditioner,
)
from src.policy.cosmos.edm import (
    EDMConfig,
    edm_scaling,
    karras_sigma_schedule,
    per_sigma_weight,
    sample_sigma,
)


@dataclass
class PolicyOutputs:
    pred_latents: torch.Tensor                          # (B, C, T_total, H, W) — full predicted sequence
    backbone_loss: Optional[torch.Tensor] = None        # scalar flow-matching MSE if next-state given
    pred_action_chunk: Optional[torch.Tensor] = None    # (B, chunk_size, ACTION_DIM) at inference
    pred_value: Optional[torch.Tensor] = None           # (B,) at inference


@dataclass
class LossOutputs:
    """Joint flow-matching loss broken out by latent-sequence component.

    Each component (`world` over image latents, `action` over the action latent,
    `value` over the value latent) is computed as a per-component MSE — already
    normalized by that component's element count, so the three terms are on the
    same scale regardless of T_img. `total` is `Σ λ_i · loss_i`.
    """

    total: torch.Tensor
    world: torch.Tensor
    action: torch.Tensor
    value: Optional[torch.Tensor] = None

    def detach_dict(self) -> dict:
        out = {"total": float(self.total.detach()), "world": float(self.world.detach()),
               "action": float(self.action.detach())}
        if self.value is not None:
            out["value"] = float(self.value.detach())
        return out


class BackboneAdapter(Protocol):
    """Calls the backbone transformer."""

    def __call__(
        self,
        transformer: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        crossattn_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


class DiffusersStyleAdapter:
    def __call__(self, transformer, x, timestep, crossattn_emb, padding_mask=None):
        if timestep.dim() == 2:
            timestep = timestep[:, 0]
        out = transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=crossattn_emb,
            encoder_attention_mask=padding_mask,
            return_dict=True,
        )
        return out.sample if hasattr(out, "sample") else out[0]


class CosmosNativeAdapter:
    def __call__(self, transformer, x, timestep, crossattn_emb, padding_mask=None):
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(-1, x.shape[2])
        out = transformer(
            x_B_C_T_H_W=x,
            timesteps_B_T=timestep,
            crossattn_emb=crossattn_emb,
            padding_mask=padding_mask,
        )
        return out[0] if isinstance(out, (tuple, list)) else out


def make_adapter(style: str = "diffusers") -> BackboneAdapter:
    if style == "diffusers":
        return DiffusersStyleAdapter()
    if style == "cosmos_native":
        return CosmosNativeAdapter()
    raise ValueError(f"unknown adapter style: {style}")


class CosmosWorldActionPolicy(nn.Module):
    """Goal-conditioned world-action policy on top of Cosmos-Predict2.5.

    Args:
      transformer: the Cosmos backbone transformer (from `pipe.transformer`).
      latent_channels: number of channels in the VAE latent (16 for Cosmos).
      goal_dim: dimension of the normalized goal vector (default 8).
      crossattn_dim: cross-attention emb dim — matches Cosmos T5-11B (1024) by default.
      n_goal_tokens: cross-attention tokens per goal (default 4).
      action_dim: per-step action dim (5 for our (Δx, Δy, Δz, Δyaw, Δpitch)).
      chunk_size: number of future action steps predicted per diffusion sample.
                  1 for v6 single-shot prototyping; bump to 4 / 8 once
                  trajectory data (issue #17) lands.
      use_value_latent: append a value latent frame after the action latent.
                       True by default per issue #18.
      freeze_backbone: if True (default for prototype), only train conditioner.
      adapter: which call signature to use for the backbone.
      anchor_path: path to the T5 anchor file (see scripts/build_t5_anchor.py).
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        latent_channels: int = 16,
        goal_dim: int = 8,
        crossattn_dim: int = COSMOS_CROSSATTN_DIM,
        n_goal_tokens: int = 4,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 1,
        use_value_latent: bool = True,
        freeze_backbone: bool = True,
        adapter: BackboneAdapter | str = "diffusers",
        anchor_path: str | None = None,
        lambda_world: float = 1.0,
        lambda_action: float = 1.0,
        lambda_value: float = 1.0,
        edm_config: EDMConfig | None = None,
        action_scale: "np.ndarray | torch.Tensor | None" = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.adapter = make_adapter(adapter) if isinstance(adapter, str) else adapter
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_value_latent = use_value_latent
        self.lambda_world = lambda_world
        self.lambda_action = lambda_action
        self.lambda_value = lambda_value
        self.edm = edm_config or EDMConfig()

        # Per-dim action normalization scale, registered as a buffer so it travels
        # with the checkpoint — the dataset normalizes by this and sample() can
        # denormalize by the same values, so train/eval can never drift even if
        # the module-level ACTION_SCALE constant later changes.
        scale = ACTION_SCALE if action_scale is None else action_scale
        self.register_buffer("action_scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))

        self.conditioner = ShotProfileVectorConditioner(
            goal_dim=goal_dim,
            model_dim=crossattn_dim,
            n_tokens=n_goal_tokens,
            anchor_path=anchor_path,
        )

        if freeze_backbone:
            for p in self.transformer.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Latent-sequence assembly helpers
    # ------------------------------------------------------------------

    def num_extra_latent_frames(self) -> int:
        """Number of latent frames appended after the image latents."""
        return 1 + (1 if self.use_value_latent else 0)

    def action_latent_idx(self, t_img: int) -> int:
        """Index of the action latent frame in the full sequence (= last image idx + 1)."""
        return t_img

    def value_latent_idx(self, t_img: int) -> int:
        if not self.use_value_latent:
            raise ValueError("value latent disabled")
        return t_img + 1

    def build_training_latents(
        self,
        image_latent: torch.Tensor,                  # (B, C, T_img, H, W)
        action_chunk: torch.Tensor,                  # (B, chunk_size, action_dim)
        value_target: Optional[torch.Tensor] = None, # (B,)
    ) -> torch.Tensor:
        """Assemble the clean latent sequence x0 with action+value injected.

        Returns: (B, C, T_total, H, W)
        """
        b, c, t_img, h, w = image_latent.shape
        n_extra = self.num_extra_latent_frames()
        x0 = torch.zeros(b, c, t_img + n_extra, h, w, dtype=image_latent.dtype, device=image_latent.device)
        x0[:, :, :t_img] = image_latent

        a_idx = self.action_latent_idx(t_img)
        x0[:, :, a_idx] = inject_action_chunk(x0[:, :, a_idx], action_chunk)

        if self.use_value_latent:
            v_idx = self.value_latent_idx(t_img)
            if value_target is None:
                value_target = torch.zeros(b, dtype=image_latent.dtype, device=image_latent.device)
            x0[:, :, v_idx] = inject_value(x0[:, :, v_idx], value_target)

        return x0

    # ------------------------------------------------------------------
    # Training forward — returns the flow-matching loss
    # ------------------------------------------------------------------

    def _backbone_x0_pred(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        cond: ShotProfileCondition,
    ) -> torch.Tensor:
        """Run the backbone under EDM preconditioning and return the x0 prediction.

        Per Karras 2022 eq. 7 (also `_src/predict2/models/text2world_model.py:910`):
            x_t_in     = c_in · x_t
            timestep   = c_noise = 0.25 · log(σ)
            net_out    = transformer(x_t_in, timestep, cond)
            x0_pred    = c_skip · x_t + c_out · net_out
        """
        b = x_t.shape[0]
        c_skip, c_out, c_in, c_noise = edm_scaling(sigma, self.edm.sigma_data)
        view_shape = (b,) + (1,) * (x_t.dim() - 1)
        c_skip_v = c_skip.view(*view_shape).to(x_t.dtype)
        c_out_v = c_out.view(*view_shape).to(x_t.dtype)
        c_in_v = c_in.view(*view_shape).to(x_t.dtype)
        c_noise_v = c_noise.to(x_t.dtype)

        net_out = self.adapter(
            self.transformer,
            x_t * c_in_v,
            c_noise_v,
            cond.crossattn_emb,
            cond.padding_mask,
        )
        return c_skip_v * x_t + c_out_v * net_out

    def compute_loss(
        self,
        image_latent: torch.Tensor,                  # (B, C, T_img, H, W)
        action_chunk: torch.Tensor,                  # (B, chunk_size, action_dim)
        goal_vec: torch.Tensor,                      # (B, goal_dim)
        value_target: Optional[torch.Tensor] = None, # (B,)
    ) -> LossOutputs:
        """Joint EDM x0-prediction loss decomposed by latent-sequence component.

        Pipeline (Karras 2022 + cosmos-policy `compute_loss_with_epsilon_and_sigma`):
          1. Sample σ from log-normal `N(p_mean, p_std)`, then apply
             `BALANCED_TWO_HEADS_V1` if enabled (forces some samples to σ extremes).
          2. Form `x_t = x0 + σ · ε`.
          3. Run backbone under EDM preconditioning → `x0_pred`.
          4. Per-σ-weighted squared error `(x0 - x0_pred)² · w(σ)`.
          5. Split by latent-sequence position → world / action / value components.

        Returns `LossOutputs(total, world, action, value)` with components on a
        comparable scale (per-element means) so `λ` weights are intuitive.
        """
        x0 = self.build_training_latents(image_latent, action_chunk, value_target)
        cond = self.conditioner({"goal_vec": goal_vec})

        b = x0.shape[0]
        sigma = sample_sigma(b, self.edm, device=x0.device).to(x0.dtype)   # (B,)
        epsilon = torch.randn_like(x0)
        view_shape = (b,) + (1,) * (x0.dim() - 1)
        sigma_v = sigma.view(*view_shape)

        x_t = x0 + sigma_v * epsilon
        x0_pred = self._backbone_x0_pred(x_t, sigma, cond)

        weights = per_sigma_weight(sigma, self.edm.sigma_data, scaling="edm").to(x0.dtype)
        sq = ((x0_pred - x0) ** 2) * weights.view(*view_shape)              # (B, C, T_total, H, W)

        t_img = image_latent.shape[2]
        a_idx = self.action_latent_idx(t_img)
        loss_world = sq[:, :, :t_img].mean()
        loss_action = sq[:, :, a_idx].mean()
        if self.use_value_latent:
            v_idx = self.value_latent_idx(t_img)
            loss_value = sq[:, :, v_idx].mean()
            total = (
                self.lambda_world * loss_world
                + self.lambda_action * loss_action
                + self.lambda_value * loss_value
            )
            return LossOutputs(total=total, world=loss_world, action=loss_action, value=loss_value)

        total = self.lambda_world * loss_world + self.lambda_action * loss_action
        return LossOutputs(total=total, world=loss_world, action=loss_action, value=None)

    # ------------------------------------------------------------------
    # Inference — diffusion sample, then extract action + value
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        image_latent: torch.Tensor,                  # (B, C, T_img, H, W) — conditioning frames
        goal_vec: torch.Tensor,                      # (B, goal_dim)
        *,
        n_steps: int = 32,
        denormalize: bool = True,
    ) -> PolicyOutputs:
        """EDM sampling with the Karras 2022 σ schedule + Euler step.

        Initialize with noise at σ_max, denoise step-by-step down to σ_min, pin
        the image-conditioning frames at every step. The action + value latent
        positions are sampled by the diffusion process.

        `denormalize=True` (default) returns the action chunk in physical units
        (metres / radians) by multiplying through `self.action_scale` — the same
        buffer the dataset normalized by — so callers never need to know the scale
        and train/eval can't drift. Pass `denormalize=False` to get the raw
        normalized action.
        """
        b, c, t_img, h, w = image_latent.shape
        n_extra = self.num_extra_latent_frames()
        t_total = t_img + n_extra
        cond = self.conditioner({"goal_vec": goal_vec})

        sigmas = karras_sigma_schedule(n_steps, self.edm, device=image_latent.device)
        # Start from pure noise at σ_max
        x = torch.randn(b, c, t_total, h, w, dtype=image_latent.dtype, device=image_latent.device)
        x = x * sigmas[0].to(x.dtype)
        x[:, :, :t_img] = image_latent                          # pin conditioning frames

        view_shape = (b,) + (1,) * (x.dim() - 1)
        for i in range(n_steps):
            sigma_i = sigmas[i].to(x.dtype).expand(b)
            sigma_next = sigmas[i + 1].to(x.dtype).expand(b)
            x0_pred = self._backbone_x0_pred(x, sigma_i, cond)
            # Euler step: d = (x - x0_pred) / σ; x_next = x + (σ_next - σ) · d
            d = (x - x0_pred) / sigma_i.view(*view_shape).clamp_min(1e-8)
            x = x + (sigma_next - sigma_i).view(*view_shape) * d
            x[:, :, :t_img] = image_latent                      # re-pin image frames each step

        a_idx = self.action_latent_idx(t_img)
        pred_action = extract_action_chunk(
            x[:, :, a_idx], chunk_size=self.chunk_size, action_dim=self.action_dim,
        )
        if denormalize:
            pred_action = pred_action * self.action_scale.to(pred_action.dtype)
        pred_value: Optional[torch.Tensor] = None
        if self.use_value_latent:
            v_idx = self.value_latent_idx(t_img)
            pred_value = extract_value(x[:, :, v_idx])

        return PolicyOutputs(pred_latents=x, pred_action_chunk=pred_action, pred_value=pred_value)


__all__ = [
    "CosmosWorldActionPolicy",
    "PolicyOutputs",
    "LossOutputs",
    "BackboneAdapter",
    "DiffusersStyleAdapter",
    "CosmosNativeAdapter",
    "make_adapter",
]
