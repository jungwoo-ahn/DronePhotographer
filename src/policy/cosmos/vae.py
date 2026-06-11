"""Wrapper around the Cosmos-Predict2.5 VAE.

Verified against upstream `cosmos_policy/_src/predict2/tokenizers/base_vae.py`:

  - **Axis convention**: `(B, C, T, H, W)` everywhere — same on input and on the
    latent. No `(B, T, C, H, W)` rearrangement.
  - **Encoded output**: a plain `torch.Tensor` of shape `(B, 16, T_latent, H_latent, W_latent)`
    already rescaled by `(latent_mean, latent_std)`. There is **no `latent_dist`**
    (unlike `diffusers.AutoencoderKL`).
  - **Image input range**: `[-1, 1]` after `data / 127.5 - 1.0` normalization
    (Cosmos applies this in-place before encode).
  - **Temporal compression**: not a fixed 4-frame constraint — it depends on the
    tokenizer (Wan2pt1 / Wan2pt2). For our prototype we assemble a 4-frame clip
    `[state, next_state, pad, pad]` so the temporal axis is non-trivial.

The wrapper handles both:
  - Native Cosmos VAEs (direct tensor return).
  - Diffusers-wrapped VAEs (`.latent_dist.sample()` style), in case
    `DiffusionPipeline.from_pretrained` returns an `AutoencoderKL`-style object.

We keep the wrapper minimal — the heavy lifting is in the upstream VAE.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

VAE_LATENT_CHANNELS = 16  # cosmos-policy uses 16-channel latents (predict2)


class CosmosVAEWrapper(nn.Module):
    """Wraps the VAE pulled from `DiffusionPipeline.from_pretrained("nvidia/Cosmos-Predict2.5-2B").vae`."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        cfg = getattr(vae, "config", None)
        # AutoencoderKLWan (the Cosmos-2.5 diffusers VAE) normalizes with per-channel
        # latents_mean / latents_std; plain AutoencoderKL uses a scalar scaling_factor.
        self.scaling_factor: Optional[float] = getattr(cfg, "scaling_factor", None) if cfg else None
        lm = getattr(cfg, "latents_mean", None) if cfg else None
        ls = getattr(cfg, "latents_std", None) if cfg else None
        if lm is not None and ls is not None:
            self.register_buffer("latents_mean", torch.tensor(lm).view(1, -1, 1, 1, 1))
            self.register_buffer("latents_std", torch.tensor(ls).view(1, -1, 1, 1, 1))
        else:
            self.latents_mean = None
            self.latents_std = None

    @classmethod
    def from_pretrained(cls, repo_id: str = "nvidia/Cosmos-Predict2.5-2B", **kwargs) -> "CosmosVAEWrapper":
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(repo_id, **kwargs)
        return cls(pipe.vae)

    @staticmethod
    def assemble_clip(state_image: torch.Tensor, next_state_image: torch.Tensor, *, T: int = 4) -> torch.Tensor:
        """Build a (B, C, T, H, W) clip from two single-frame tensors.

        Layout: frame 0 = state, frame 1 = next_state, frames 2..T-1 = next_state
        (repeated to fill the temporal axis without injecting zero frames that
        would shift the VAE's statistics). Once issue #17 lands real
        trajectories, this helper goes away and the dataset emits a full clip.
        """
        if state_image.shape != next_state_image.shape:
            raise ValueError("state and next_state must have the same shape")
        # state_image / next_state_image: (B, C, H, W)
        b, c, h, w = state_image.shape
        clip = torch.empty(b, c, T, h, w, dtype=state_image.dtype, device=state_image.device)
        clip[:, :, 0] = state_image
        for t in range(1, T):
            clip[:, :, t] = next_state_image
        return clip

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a `(B, C, T, H, W)` video clip in [-1, 1] to a latent.

        Returns: `(B, 16, T_latent, H_latent, W_latent)` — already rescaled.
        """
        out = self.vae.encode(video)
        # Cosmos native: direct tensor; diffusers: object with `.latent_dist`
        if hasattr(out, "latent_dist"):
            latent = out.latent_dist.sample()
            return self._normalize(latent)
        if isinstance(out, torch.Tensor):
            return out
        # Some diffusers VAEs return a dict-like object with a `latents` field
        if hasattr(out, "latents"):
            return self._normalize(out.latents)
        raise TypeError(f"unexpected VAE.encode output type: {type(out)}")

    def _normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE space -> model latent space (inverse of the pipeline's pre-decode step)."""
        if self.latents_mean is not None:
            mean = self.latents_mean.to(latent.device, latent.dtype)
            std = self.latents_std.to(latent.device, latent.dtype)
            return (latent - mean) / std
        if self.scaling_factor is not None:
            return latent * self.scaling_factor
        return latent

    def _denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latents_mean is not None:
            mean = self.latents_mean.to(latent.device, latent.dtype)
            std = self.latents_std.to(latent.device, latent.dtype)
            return latent * std + mean
        if self.scaling_factor is not None:
            return latent / self.scaling_factor
        return latent

    def encode_frame(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image `(B, C, H, W)` as a T=1 clip -> `(B, 16, 1, h, w)`.

        The Wan VAE requires (T - 1) % 4 == 0; T=1 is the image special case.
        Encoding state and goal frames separately (then concatenating latents)
        gives exactly two clean latent frames — ALOHA-style T_img=2 — with no
        temporal grouping blur.
        """
        return self.encode(image.unsqueeze(2))

    def encode_pair_frames(self, state_image: torch.Tensor, next_state_image: torch.Tensor) -> torch.Tensor:
        """Encode (state, goal) images separately and concat: `(B, 16, 2, h, w)`."""
        return torch.cat([self.encode_frame(state_image), self.encode_frame(next_state_image)], dim=2)

    def encode_pair(self, state_image: torch.Tensor, next_state_image: torch.Tensor, *, T: int = 4) -> torch.Tensor:
        """Convenience: assemble a 4-frame clip from (state, next_state) and encode."""
        clip = self.assemble_clip(state_image, next_state_image, T=T)
        return self.encode(clip)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode `(B, 16, T_latent, H_latent, W_latent)` back to `(B, C, T, H, W)` in [-1, 1]."""
        out = self.vae.decode(self._denormalize(latent))
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"unexpected VAE.decode output type: {type(out)}")
