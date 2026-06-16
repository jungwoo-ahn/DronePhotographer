"""ConditionalUnet1D — the canonical Diffusion Policy CNN denoiser (Chi et al.).

A 1D U-Net over the *temporal* (action-chunk) axis: it denoises a noised action
chunk `(B, T, action_dim)` conditioned globally (FiLM) on a single vector that
carries both the diffusion-timestep embedding and the observation/goal embedding.
This is the "obs as global conditioning" variant from the Diffusion Policy paper
(its recommended CNN backbone), reimplemented compactly against our infra. See
REFERENCES.md.

Shapes: `forward(sample (B, T, A), timestep (B,), global_cond (B, G)) -> (B, T, A)`.
Internally everything runs as `(B, C, T)` for `Conv1d`; we transpose at the seams.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding of the (scalar) diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp: int, out: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out), out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with FiLM (scale+bias) injected from the global cond."""

    def __init__(self, inp: int, out: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8) -> None:
        super().__init__()
        self.out_channels = out
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp, out, kernel_size, n_groups),
            Conv1dBlock(out, out, kernel_size, n_groups),
        ])
        # FiLM: cond -> per-channel (scale, bias) applied after the first block.
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out * 2))
        self.residual_conv = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).view(x.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """1D conditional U-Net denoiser.

    Args:
      input_dim: action dimension (channels of the sequence).
      global_cond_dim: dim of the external conditioning vector (obs + goal embed).
      diffusion_step_embed_dim: width of the timestep embedding.
      down_dims: channel widths per U-Net level (len-1 downsamples).
      kernel_size / n_groups: Conv1dBlock params.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        *,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        all_dims = [input_dim, *down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        start_dim = down_dims[0]
        mid_dim = all_dims[-1]

        rb = lambda i, o: ConditionalResidualBlock1D(i, o, cond_dim, kernel_size, n_groups)

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                rb(dim_in, dim_out), rb(dim_out, dim_out),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.mid_modules = nn.ModuleList([rb(mid_dim, mid_dim), rb(mid_dim, mid_dim)])

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                rb(dim_out * 2, dim_in), rb(dim_in, dim_in),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # (B, T, A) -> (B, A, T)
        x = sample.transpose(1, 2)
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        t = t.expand(x.shape[0]).to(x.device)
        cond = torch.cat([self.diffusion_step_encoder(t), global_cond], dim=-1)

        skips = []
        for r1, r2, down in self.down_modules:
            x = r1(x, cond)
            x = r2(x, cond)
            skips.append(x)
            x = down(x)
        for m in self.mid_modules:
            x = m(x, cond)
        for r1, r2, up in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = r1(x, cond)
            x = r2(x, cond)
            x = up(x)
        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, A, T) -> (B, T, A)


__all__ = ["ConditionalUnet1D"]
