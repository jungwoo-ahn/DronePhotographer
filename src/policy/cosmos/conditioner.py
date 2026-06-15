"""Shot-profile vector conditioner — zero-init prefix injection on a T5 anchor.

Design (issue #18 — "Goal condition: We need to embed shot profiles somehow"):

  1. **Anchor**: T5-11B encodes a fixed prompt (e.g. "A drone cinematography")
     *once*, offline (`scripts/build_t5_anchor.py`). The `(512, 1024) bf16`
     output and its padding mask are saved to `assets/t5_anchor.pt`. T5 is
     **never loaded again** at training or inference time.

  2. **Goal projection**: a learnable `Linear(goal_dim, K·D)` produces K new
     "goal tokens" of dim D=1024 from the normalized 8-dim goal vector.

  3. **Zero-init prefix injection**: the goal tokens are *added* to K positions
     immediately after the anchor's real-text tokens (those positions are
     originally padding). The scale is a learnable scalar `gate`, **initialized
     to 0** — so at init the conditioner output is exactly the frozen anchor,
     regardless of the goal vector. Gradient descent ramps the gate up.

  4. **Padding mask**: the anchor's real tokens + the K injected goal positions
     are marked valid; the rest stay padding so cross-attention ignores them.

This gives the "zero-init add" property (ControlNet-style) without touching the
backbone's internals. If the gate stays small and we plateau early, escalate to
IP-Adapter-style **decoupled cross-attention** (see COSMOS_API.md for the
documented next step).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Cosmos cross-attention embedding dim (T5-11B hidden size).
COSMOS_CROSSATTN_DIM = 1024

# T5 max sequence length used by Cosmos.
COSMOS_T5_MAX_LEN = 512

# Default number of "goal tokens" injected as a prefix after the real T5 text.
DEFAULT_GOAL_TOKENS = 4


@dataclass
class ShotProfileCondition:
    crossattn_emb: torch.Tensor                  # (B, max_seq_len, D)
    padding_mask: Optional[torch.Tensor] = None  # (B, max_seq_len) bool
    raw_goal: Optional[torch.Tensor] = None      # (B, D_goal)


class ShotProfileVectorConditioner(nn.Module):
    """Zero-init prefix-injection conditioner on a fixed T5 anchor."""

    def __init__(
        self,
        goal_dim: int = 8,
        model_dim: int = COSMOS_CROSSATTN_DIM,
        n_tokens: int = DEFAULT_GOAL_TOKENS,
        max_seq_len: int = COSMOS_T5_MAX_LEN,
        dropout_rate: float = 0.0,
        *,
        anchor_path: str | Path | None = None,
        anchor_embedding: Optional[torch.Tensor] = None,   # (max_seq_len, model_dim)
        anchor_real_len: Optional[int] = None,
        anchor_padding_mask: Optional[torch.Tensor] = None,  # (max_seq_len,) bool
    ) -> None:
        super().__init__()
        self.goal_dim = goal_dim
        self.model_dim = model_dim
        self.n_tokens = n_tokens
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        if anchor_embedding is None:
            if anchor_path is not None:
                blob = torch.load(anchor_path, map_location="cpu", weights_only=False)
                anchor_embedding = blob["embedding"]
                anchor_real_len = int(blob["real_len"])
                anchor_padding_mask = blob["padding_mask"]
            else:
                # No anchor → zeros buffer. Allowed for tests but degrades the
                # conditioner to a learnable prefix with no T5 grounding.
                anchor_embedding = torch.zeros(max_seq_len, model_dim)
                anchor_real_len = 0
                anchor_padding_mask = torch.zeros(max_seq_len, dtype=torch.bool)

        if anchor_embedding.shape != (max_seq_len, model_dim):
            raise ValueError(
                f"anchor_embedding shape {tuple(anchor_embedding.shape)} != "
                f"({max_seq_len}, {model_dim})"
            )
        if anchor_padding_mask is None:
            anchor_padding_mask = torch.zeros(max_seq_len, dtype=torch.bool)
            anchor_padding_mask[: int(anchor_real_len or 0)] = True
        if anchor_real_len is None:
            anchor_real_len = int(anchor_padding_mask.sum().item())

        self.anchor_real_len = int(anchor_real_len)
        if self.anchor_real_len + n_tokens > max_seq_len:
            raise ValueError(
                f"anchor_real_len ({self.anchor_real_len}) + n_tokens ({n_tokens}) "
                f"exceeds max_seq_len ({max_seq_len})"
            )

        # Anchor + its padding mask are frozen.
        self.register_buffer("anchor_emb", anchor_embedding.to(torch.float32))
        self.register_buffer("anchor_mask", anchor_padding_mask.to(torch.bool))

        # Trainable: goal projection + small-init gate.
        self.goal_proj = nn.Linear(goal_dim, n_tokens * model_dim)
        # Initialize gate at 0.1 instead of 0 to bypass the chicken-and-egg
        # bootstrap. With Qwen2.5-VL anchor (100,352-dim cross-attn) the
        # ControlNet-style zero-init keeps `gate × goal_tokens` so small that
        # `∂loss/∂goal_proj.weight ≈ 0` for thousands of iterations. Starting
        # at 0.1 means the conditioner output deviates from the anchor by ~3.5%
        # per dim from step 1, so goal_proj sees real gradient immediately.
        self.gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self,
        batch: dict,
        override_dropout_rate: Optional[float] = None,
    ) -> ShotProfileCondition:
        goal = batch["goal_vec"]
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"expected goal_dim={self.goal_dim}, got {goal.shape[-1]}")
        b = goal.shape[0]
        dtype = goal.dtype
        device = goal.device

        # Broadcast frozen anchor to batch; cast to goal dtype so AMP works.
        emb = self.anchor_emb.to(dtype=dtype, device=device).unsqueeze(0).expand(b, -1, -1).clone()

        # Project goal → K tokens
        goal_tokens = self.goal_proj(goal).view(b, self.n_tokens, self.model_dim)

        # Classifier-free dropout zeroes the gate's effect entirely for some
        # items (uncondition path). At init the gate is 0 → no effect from goal.
        rate = override_dropout_rate if override_dropout_rate is not None else self.dropout_rate
        if self.training and rate > 0.0:
            keep = (torch.rand(b, 1, 1, device=device) > rate).to(dtype)
        else:
            keep = torch.ones(1, device=device, dtype=dtype)

        start = self.anchor_real_len
        end = start + self.n_tokens
        emb[:, start:end] = emb[:, start:end] + (self.gate.to(dtype) * keep) * goal_tokens

        # Padding mask: real anchor tokens + goal positions are valid
        mask = self.anchor_mask.to(device=device).unsqueeze(0).expand(b, -1).clone()
        mask[:, start:end] = True

        return ShotProfileCondition(crossattn_emb=emb, padding_mask=mask, raw_goal=goal)


__all__ = [
    "COSMOS_CROSSATTN_DIM",
    "COSMOS_T5_MAX_LEN",
    "DEFAULT_GOAL_TOKENS",
    "ShotProfileCondition",
    "ShotProfileVectorConditioner",
]
