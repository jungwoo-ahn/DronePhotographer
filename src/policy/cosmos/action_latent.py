"""Action / value injection into and extraction from latent frames.

Ported from upstream:
  - inject  ←  `cosmos_policy/models/policy_text2world_model.py::replace_latent_with_action_chunk`
  - extract ←  `cosmos_policy/experiments/robot/cosmos_utils.py::extract_action_chunk_from_latent_sequence`

There are **no learned parameters here**. The encoding is a pure tile-and-reshape
that fills a `(B, C, H, W)` latent frame with the flattened `(chunk_size, D_act)`
action chunk, repeated as many times as fits. The decoder splits the latent into
those repeated tiles and averages them for stability (robust to per-element noise
in the diffusion sample).

Value frames work the same way with a scalar — the value latent frame is just
broadcast-filled at injection and mean-reduced at extraction.
"""

from __future__ import annotations

import torch


def inject_action_chunk(latent_frame: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
    """Fill a latent frame with a tiled flat copy of the action chunk.

    Args:
      latent_frame: `(B, C, H, W)` — target frame, will be overwritten.
      action_chunk: `(B, chunk_size, action_dim)` — ground-truth actions to inject.

    Returns:
      `(B, C, H, W)` latent frame with the tiled action chunk values.
    """
    b, c, h, w = latent_frame.shape
    flat_action = action_chunk.reshape(b, -1)            # (B, chunk·act_dim)
    n_action = flat_action.shape[1]
    n_latent = c * h * w
    if n_action > n_latent:
        raise ValueError(
            f"action volume {n_action} > latent volume {n_latent}; "
            f"either lower chunk_size or use a bigger latent frame"
        )
    n_repeats = (n_latent + n_action - 1) // n_action     # ceiling
    tiled = flat_action.repeat(1, n_repeats)[:, :n_latent]
    return tiled.reshape(b, c, h, w).to(latent_frame.dtype)


def extract_action_chunk(
    latent_frame: torch.Tensor,
    *,
    chunk_size: int,
    action_dim: int,
) -> torch.Tensor:
    """Decode a latent frame back to an `(B, chunk_size, action_dim)` action chunk.

    Splits the flattened frame into all whole repeats of `(chunk_size · action_dim)`
    and averages them — the diffusion sampler produces noisy latents, but the
    repeated structure imposed at training time gives many estimates of the same
    chunk to average over.
    """
    b, c, h, w = latent_frame.shape
    n_latent = c * h * w
    n_action = chunk_size * action_dim
    if n_action > n_latent:
        raise ValueError(f"action volume {n_action} > latent volume {n_latent}")
    n_repeats = n_latent // n_action
    flat = latent_frame.reshape(b, -1)
    repeats = flat[:, : n_repeats * n_action].reshape(b, n_repeats, n_action)
    chunks = repeats.reshape(b, n_repeats, chunk_size, action_dim)
    return chunks.mean(dim=1)                              # (B, chunk_size, action_dim)


def inject_value(latent_frame: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Fill a latent frame with a scalar value broadcast across all elements.

    Args:
      latent_frame: `(B, C, H, W)`
      value: `(B,)` or `(B, 1)`
    """
    b, c, h, w = latent_frame.shape
    v = value.view(b, 1).to(latent_frame.dtype)
    return v.expand(b, c * h * w).reshape(b, c, h, w).contiguous()


def extract_value(latent_frame: torch.Tensor) -> torch.Tensor:
    """Decode a latent frame back to a scalar by mean-pooling all elements."""
    return latent_frame.reshape(latent_frame.shape[0], -1).mean(dim=1)


def action_capacity(latent_frame_shape: tuple[int, int, int, int], chunk_size: int, action_dim: int) -> int:
    """Return the number of complete repeats of the action chunk that fit in the latent.

    `(B, C, H, W)` → `floor(C·H·W / (chunk·action_dim))`. Higher = more averaging at decode.
    """
    _, c, h, w = latent_frame_shape
    return (c * h * w) // (chunk_size * action_dim)


__all__ = [
    "inject_action_chunk",
    "extract_action_chunk",
    "inject_value",
    "extract_value",
    "action_capacity",
]
