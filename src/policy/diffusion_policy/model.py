"""DiffusionPolicy — Diffusion Policy baseline ("ours without the WAM").

A frozen modern vision backbone (DINOv2-large) encodes the current image to a
global observation embedding; the normalized shot-profile goal is embedded and
concatenated; a conditional 1D U-Net (`ConditionalUnet1D`) denoises the 5D action
chunk via DDPM (epsilon-prediction), sampled at inference with DDIM. No
future-frame prediction (the world model we ablate), no value head.

This isolates previsualization: same data / goal / action / eval as the Cosmos
world-action policy, with a frozen DINOv2 encoder + DDPM action head in place of
the video world model.

Backbone seam: `self.backbone(**obs_inputs)`; the obs embedding is the backbone's
`pooler_output` (else mean-pooled `last_hidden_state`). The real backbone is
`AutoModel.from_pretrained("facebook/dinov2-large")`; tests pass a tiny mock with
the same contract, so the whole obs -> cond -> DDPM path runs without the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.policy.common.action_repr import ACTION_DIM, ACTION_SCALE
from src.policy.diffusion_policy.denoiser import ConditionalUnet1D


@dataclass
class DPLossOutputs:
    """Mirrors the VLA/Cosmos LossOutputs so trainer logging is shared. Action-only."""

    total: torch.Tensor
    action: torch.Tensor

    def detach_dict(self) -> dict:
        return {"total": float(self.total.detach()), "action": float(self.action.detach())}


@dataclass
class DPOutputs:
    pred_action_chunk: torch.Tensor    # (B, chunk, ACTION_DIM)


class DiffusionPolicy(nn.Module):
    """Frozen-vision-backbone Diffusion Policy with a conditional 1D U-Net head.

    Args:
      backbone: a DINOv2 `AutoModel` (or mock) returning `.pooler_output` (B,D)
        and/or `.last_hidden_state` (B,L,D).
      obs_dim: backbone hidden size D (read from backbone.config if None).
      goal_dim: normalized goal vector dim (default 8).
      goal_embed_dim: width the goal vector is embedded to before conditioning.
      action_dim / chunk_size: 5D action over `chunk_size` future steps.
      down_dims / diffusion_step_embed_dim: ConditionalUnet1D size.
      num_train_timesteps / beta_schedule: DDPM schedule.
      freeze_backbone: if True (default), the backbone runs under no_grad.
      action_scale: per-dim action normalization (persisted as a buffer).
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        obs_dim: Optional[int] = None,
        goal_dim: int = 8,
        goal_embed_dim: int = 128,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 8,
        down_dims: tuple[int, ...] = (128, 256, 512),
        diffusion_step_embed_dim: int = 128,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        freeze_backbone: bool = True,
        action_scale=None,
        processor=None,
    ) -> None:
        super().__init__()
        from diffusers import DDIMScheduler, DDPMScheduler

        self.backbone = backbone
        self.processor = processor          # DINOv2 image processor (real path); None for mock
        self.freeze_backbone = freeze_backbone
        if obs_dim is None:
            cfg = getattr(backbone, "config", None)
            obs_dim = int(getattr(cfg, "hidden_size"))
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_train_timesteps = num_train_timesteps

        self.goal_embed = nn.Sequential(
            nn.Linear(goal_dim, goal_embed_dim), nn.Mish(), nn.Linear(goal_embed_dim, goal_embed_dim),
        )
        self.denoiser = ConditionalUnet1D(
            action_dim, global_cond_dim=obs_dim + goal_embed_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
        )
        sched_kw = dict(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
                        prediction_type="epsilon", clip_sample=True)
        self.scheduler_train = DDPMScheduler(**sched_kw)
        self.scheduler_infer = DDIMScheduler(**sched_kw)

        import numpy as np

        scale = ACTION_SCALE if action_scale is None else action_scale
        self.register_buffer("action_scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def prepare_inputs(self, batch: dict, device, dtype) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Build (obs_inputs, goal_vec, action_chunk) from a dataloader batch.

        Runs the DINOv2 image processor on the [-1,1] CHW images. The mock tests
        bypass this and call `compute_loss` with hand-built tensors.
        """
        from src.policy.diffusion_policy.dataset import build_obs_inputs

        proc = build_obs_inputs(self.processor, batch["state_image"])
        obs_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
        goal = batch["goal_vec"].to(device, dtype)
        action = batch["action_chunk"].to(device, dtype)
        return obs_inputs, goal, action

    def global_cond(self, obs_inputs: dict, goal_vec: torch.Tensor) -> torch.Tensor:
        """Encode image (frozen) + goal -> global conditioning vector (B, obs+goal)."""
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(**obs_inputs)
        else:
            out = self.backbone(**obs_inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            pooled = h.mean(dim=1)
        goal_e = self.goal_embed(goal_vec.to(self.goal_embed[0].weight.dtype))
        return torch.cat([pooled.to(goal_e.dtype), goal_e], dim=-1)

    def compute_loss(self, obs_inputs: dict, goal_vec: torch.Tensor, action_chunk: torch.Tensor,
                     timesteps: Optional[torch.Tensor] = None) -> DPLossOutputs:
        """DDPM epsilon-prediction MSE on the action chunk.

        `timesteps` (B,) overrides the random draw — validation passes a fixed grid
        so the metric is comparable across checkpoints.
        """
        cond = self.global_cond(obs_inputs, goal_vec)
        a0 = action_chunk
        b = a0.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (b,), device=a0.device)
        timesteps = timesteps.long()
        noise = torch.randn_like(a0)
        noisy = self.scheduler_train.add_noise(a0, noise, timesteps)
        pred = self.denoiser(noisy, timesteps, cond)
        loss = (pred.float() - noise.float()).pow(2).mean()
        return DPLossOutputs(total=loss, action=loss)

    @torch.no_grad()
    def sample(self, obs_inputs: dict, goal_vec: torch.Tensor, *, n_steps: int = 16, denormalize: bool = True) -> DPOutputs:
        """DDIM-sample the action chunk (the obs/goal cond is computed once)."""
        cond = self.global_cond(obs_inputs, goal_vec)
        b = cond.shape[0]
        x = torch.randn(b, self.chunk_size, self.action_dim, device=cond.device, dtype=torch.float32)
        self.scheduler_infer.set_timesteps(n_steps, device=cond.device)
        for t in self.scheduler_infer.timesteps:
            model_out = self.denoiser(x.to(cond.dtype), t.expand(b), cond).float()
            x = self.scheduler_infer.step(model_out, t, x).prev_sample
        if denormalize:
            x = x * self.action_scale.to(x.dtype)
        return DPOutputs(pred_action_chunk=x)


__all__ = ["DiffusionPolicy", "DPLossOutputs", "DPOutputs"]
