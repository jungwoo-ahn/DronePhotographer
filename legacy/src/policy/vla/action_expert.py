"""Flow-matching action expert for the π0-style VLA baseline.

π0 augments a frozen-ish VLM with a small "action expert" that denoises a
continuous action chunk via flow matching. π0 fuses it into the VLM through
joint attention; we use the standard, less-invasive reimplementation: a small
transformer that **self-attends over the noisy action tokens** and
**cross-attends to the VLM's hidden states** (the image + goal context), and
outputs the flow velocity. See REFERENCES.md.

Input  : noisy action chunk a_sigma  (B, chunk, action_dim)
         flow sigma                  (B,)
         VLM context hidden states   (B, L, D_ctx)   [+ optional key padding mask]
Output : flow velocity v             (B, chunk, action_dim)   (target = eps - a0)
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sigma_time_embedding(sigma: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow sigma in [0, 1]. sigma: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = sigma.float()[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(sigma.dtype)


class _ExpertBlock(nn.Module):
    """Pre-norm block: self-attn over action tokens, cross-attn to VLM context, MLP."""

    def __init__(self, dim: int, ctx_dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.ctx_proj = nn.Linear(ctx_dim, dim) if ctx_dim != dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_ca(x)
        kv = self.ctx_proj(ctx)
        x = x + self.cross_attn(h, kv, kv, key_padding_mask=ctx_key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ActionExpert(nn.Module):
    """Small cross-attention flow-matching transformer over an action chunk.

    Args:
      action_dim: per-step action dim (5 for our drone action).
      ctx_dim: hidden size of the VLM context (Qwen3-VL hidden size).
      dim: action-expert width.
      depth: number of blocks.
      n_heads: attention heads.
      chunk_size: number of action steps (sequence length of the expert).
    """

    def __init__(
        self,
        action_dim: int = 5,
        ctx_dim: int = 2048,
        dim: int = 512,
        depth: int = 6,
        n_heads: int = 8,
        chunk_size: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.dim = dim
        self.in_proj = nn.Linear(action_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.sigma_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_ExpertBlock(dim, ctx_dim, n_heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, action_dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        # Zero-init the output so the expert starts as a no-op velocity (stable start).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        action_sigma: torch.Tensor,            # (B, chunk, action_dim)
        sigma: torch.Tensor,                   # (B,)
        context: torch.Tensor,                 # (B, L, ctx_dim)
        context_key_padding_mask: torch.Tensor | None = None,   # (B, L), True = pad/ignore
    ) -> torch.Tensor:
        b, chunk, _ = action_sigma.shape
        x = self.in_proj(action_sigma) + self.pos_emb[:, :chunk]
        x = x + self.sigma_mlp(sigma_time_embedding(sigma, self.dim))[:, None, :]
        for blk in self.blocks:
            x = blk(x, context, context_key_padding_mask)
        return self.out_proj(self.norm_out(x))   # (B, chunk, action_dim)


class _DiTBlock(nn.Module):
    """DiT block (GR00T-style): AdaLN-Zero timestep modulation + self-attn (action tokens)
    + cross-attn (VLM context) + MLP. The flow timestep conditions the block via adaptive
    LayerNorm (scale/shift) and a residual gate, rather than being added to the tokens once.
    """

    def __init__(self, dim: int, ctx_dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False)
        self.ctx_proj = nn.Linear(ctx_dim, dim) if ctx_dim != dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)   # AdaLN-Zero: block starts as identity
        nn.init.zeros_(self.adaLN[-1].bias)

    @staticmethod
    def _mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: torch.Tensor,
                ctx_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        s_sh, s_sc, s_g, c_sh, c_sc, c_g, m_sh, m_sc, m_g = self.adaLN(cond).chunk(9, dim=-1)
        h = self._mod(self.norm_sa(x), s_sh, s_sc)
        x = x + s_g.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.ctx_proj(ctx)
        h = self._mod(self.norm_ca(x), c_sh, c_sc)
        x = x + c_g.unsqueeze(1) * self.cross_attn(h, kv, kv, key_padding_mask=ctx_key_padding_mask, need_weights=False)[0]
        h = self._mod(self.norm_mlp(x), m_sh, m_sc)
        x = x + m_g.unsqueeze(1) * self.mlp(h)
        return x


class DiTActionExpert(nn.Module):
    """GR00T N1.5-style diffusion-transformer action head (AdaLN-Zero DiT + cross-attn to VLM).

    Drop-in for ActionExpert (same forward signature -> velocity). The flow timestep
    conditions every block via AdaLN instead of a one-shot token add — the core of GR00T's
    action head — which tends to fit multimodal action chunks more sharply.
    """

    def __init__(self, action_dim: int, ctx_dim: int = 2048, dim: int = 512, depth: int = 6,
                 n_heads: int = 8, chunk_size: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(action_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_DiTBlock(dim, ctx_dim, n_heads, mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_proj = nn.Linear(dim, action_dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.adaLN_out[-1].weight); nn.init.zeros_(self.adaLN_out[-1].bias)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)   # start as no-op velocity

    def forward(self, action_sigma: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor,
                context_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, chunk, _ = action_sigma.shape
        cond = self.cond_mlp(sigma_time_embedding(sigma, self.dim))       # (B, dim)
        x = self.in_proj(action_sigma) + self.pos_emb[:, :chunk]
        for blk in self.blocks:
            x = blk(x, context, cond, context_key_padding_mask)
        sh, sc = self.adaLN_out(cond).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        return self.out_proj(x)


__all__ = ["ActionExpert", "DiTActionExpert", "sigma_time_embedding"]
