"""Shot-profile vector conditioner — conditional prefix tuning on a Qwen text anchor.

Design (issue #18 — "Goal condition: We need to embed shot profiles somehow"):

  1. **Anchor**: Cosmos-Predict2.5's text encoder — **Qwen2.5-VL (Cosmos-Reason1),
     NOT T5** — encodes a fixed prompt (e.g. "A drone cinematography") *once*,
     offline (`scripts/build_text_anchor.py`). The saved embedding is the
     per-position-normalized concatenation of all Qwen layers
     (28 × 3584 = 100352 dims/token). The real cross-attention dim is detected
     from the backbone at build time (see `train_cosmos_policy.py`); the module
     default `COSMOS_CROSSATTN_DIM` below is only a small fallback for tests.

  2. **Goal projection**: a learnable `Linear(goal_dim, K·D)` produces K continuous
     "goal tokens" of dim D from the normalized 8-dim goal vector.

  3. **Conditional prefix tuning (magnitude-matched)**: the K goal tokens are
     **concatenated** after the anchor's real text tokens and the backbone attends to
     them — standard prefix tuning, prefix conditioned on the goal. The goal tokens are
     **RMSNorm'd to the anchor's per-token scale** (per-element unit RMS → L2 ≈
     sqrt(model_dim)) times a **learnable gain** (init 1 = anchor parity). This is the
     fix for the goal being inaudible: the saved anchor is per-position-normalized
     (per-token L2 ≈ 317 at model_dim=100352), so a **zero-init** `goal_proj` (the old
     design) left the goal tokens ~1700× quieter than the anchor and they never ramped
     out of it (measured goal-dependence ratio ~0.02). Matching the magnitude by
     construction makes the goal audible from step 1; `goal_proj` then only learns token
     *direction* (content), and the gain lets training dial loudness. No zero-init
     deadlock (the failure mode of the earlier scalar `gate`, which we also dropped).

  4. **No padding tokens are ever emitted.** We pass ONLY the anchor's real text
     tokens + the K goal tokens to cross-attention. The previous design kept all
     512 anchor positions (real text + ~500 *non-zero*, per-position-normalized
     padding embeddings) and relied on a mask that the backbone adapter never
     applied — so the goal was drowned by ~500 constant padding tokens. Emitting
     only valid tokens removes both the masking requirement and that dilution.

This gives the "zero-init add" property (ControlNet-style) without touching the
backbone's internals. If the gate stays small and we plateau early, escalate to
IP-Adapter-style **decoupled cross-attention** (see COSMOS_API.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Fallback cross-attention dim for tests / naive construction. The REAL Cosmos
# dim (Qwen2.5-VL per-layer concat = 3584 × 28 = 100352) is detected from the
# backbone in train_cosmos_policy.py and passed in explicitly.
COSMOS_CROSSATTN_DIM = 1024

# Max anchor length we accept (the saved anchor may be padded to this; we only
# ever read its first `real_len` tokens).
COSMOS_TEXT_MAX_LEN = 512

# Default number of "goal tokens" concatenated after the real text tokens.
DEFAULT_GOAL_TOKENS = 4


@dataclass
class ShotProfileCondition:
    crossattn_emb: torch.Tensor                  # (B, real_len + K, D) — all valid tokens
    padding_mask: Optional[torch.Tensor] = None  # (B, real_len + K) bool, all True (no padding)
    raw_goal: Optional[torch.Tensor] = None      # (B, D_goal)


class ShotProfileVectorConditioner(nn.Module):
    """Goal-token conditioner on a fixed Qwen text anchor: goal tokens are RMSNorm'd to
    the anchor's per-token magnitude × a learnable gain (no padding emitted)."""

    def __init__(
        self,
        goal_dim: int = 8,
        model_dim: int = COSMOS_CROSSATTN_DIM,
        n_tokens: int = DEFAULT_GOAL_TOKENS,
        max_seq_len: int = COSMOS_TEXT_MAX_LEN,
        dropout_rate: float = 0.0,
        *,
        anchor_path: str | Path | None = None,
        anchor_embedding: Optional[torch.Tensor] = None,   # (L, model_dim), L >= real_len
        anchor_real_len: Optional[int] = None,
        anchor_padding_mask: Optional[torch.Tensor] = None,  # (L,) bool (real tokens True)
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
                anchor_padding_mask = blob.get("padding_mask")
            else:
                # No anchor → empty text. Allowed for tests but degrades the
                # conditioner to a learnable prefix with no Qwen grounding.
                anchor_embedding = torch.zeros(0, model_dim)
                anchor_real_len = 0
                anchor_padding_mask = None

        if anchor_embedding.dim() != 2 or anchor_embedding.shape[1] != model_dim:
            raise ValueError(
                f"anchor_embedding shape {tuple(anchor_embedding.shape)} != (L, {model_dim})"
            )
        if anchor_real_len is None:
            anchor_real_len = (int(anchor_padding_mask.sum().item())
                               if anchor_padding_mask is not None else anchor_embedding.shape[0])
        if anchor_real_len > anchor_embedding.shape[0]:
            raise ValueError(
                f"anchor_real_len ({anchor_real_len}) > anchor length ({anchor_embedding.shape[0]})"
            )
        self.anchor_real_len = int(anchor_real_len)

        # Store ONLY the real text tokens — the padding region is never used.
        self.register_buffer("anchor_text", anchor_embedding[: self.anchor_real_len].to(torch.float32))

        # Trainable goal projection (conditional prefix tuning). It learns the goal
        # token DIRECTION (content); the MAGNITUDE is set structurally in forward()
        # (RMSNorm to the anchor's per-token scale × a learnable gain), so it is NOT
        # zero-initialized. History: a zero-init proj against a fixed per-position-
        # normalized anchor (per-token L2 = sqrt(model_dim) ≈ 317) left the goal tokens
        # at ~0.06% of the anchor magnitude — inaudible in cross-attention (goal-
        # dependence ratio ~0.02) and unable to ramp out of it under the weak, noise-
        # dominated action-frame gradient. Matching the magnitude by construction
        # removes that failure mode.
        self.goal_proj = nn.Linear(goal_dim, n_tokens * model_dim)
        if n_tokens > 0:
            # Scalar loudness. Init 1.0 → after the RMSNorm in forward() each goal
            # token starts at per-element unit RMS (L2 ≈ sqrt(model_dim)), i.e. at
            # parity with the anchor tokens. Learnable, so training can dial it (down
            # to ~0 if the goal truly doesn't help) — no zero-init deadlock.
            self.goal_gain = nn.Parameter(torch.ones(()))

    def forward(
        self,
        batch: dict,
        override_dropout_rate: Optional[float] = None,
        force_uncond: bool = False,
    ) -> ShotProfileCondition:
        goal = batch["goal_vec"]
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"expected goal_dim={self.goal_dim}, got {goal.shape[-1]}")
        b = goal.shape[0]
        dtype = goal.dtype
        device = goal.device

        # Real anchor text tokens only (never the padding region).
        text = self.anchor_text.to(dtype=dtype, device=device).unsqueeze(0).expand(b, -1, -1)  # (B, real_len, D)

        # Project goal → K prefix tokens, then match the anchor's per-token magnitude:
        # RMSNorm each token to unit per-element RMS (→ L2 ≈ sqrt(model_dim), the
        # per-position-normalized anchor scale) × a learnable gain. Without this the
        # goal tokens sit ~1700× below the anchor and cross-attention can't hear them.
        # Normed in fp32 (reducing over model_dim ~1e5 in bf16 is lossy); zero input
        # stays zero (eps floor), so a genuinely all-zero goal can't produce NaNs.
        goal_tokens = self.goal_proj(goal).view(b, self.n_tokens, self.model_dim)
        if self.n_tokens > 0:
            g = goal_tokens.float()
            g = g * torch.rsqrt(g.pow(2).mean(dim=-1, keepdim=True) + 1e-6)   # per-token unit RMS
            goal_tokens = (g * self.goal_gain.float()).to(dtype)

        # Classifier-free conditioning. `force_uncond` (inference) zeros EVERY
        # goal token — the unconditional pass for guided sampling, reproducing
        # what training dropout emits (eval mode makes the dropout branch below
        # inert, so guidance needs this explicit path). Otherwise, during
        # training, `dropout_rate` zeros the goal for a random subset of items so
        # the model also learns the unconditional score P(x|∅) needed for CFG.
        if force_uncond:
            goal_tokens = torch.zeros_like(goal_tokens)
        else:
            rate = override_dropout_rate if override_dropout_rate is not None else self.dropout_rate
            if self.training and rate > 0.0:
                keep = (torch.rand(b, 1, 1, device=device) > rate).to(dtype)
                goal_tokens = keep * goal_tokens

        # Concatenate: [real text | goal tokens]. Every emitted token is valid.
        emb = torch.cat([text, goal_tokens], dim=1)                      # (B, real_len + K, D)
        mask = torch.ones(b, emb.shape[1], dtype=torch.bool, device=device)
        return ShotProfileCondition(crossattn_emb=emb, padding_mask=mask, raw_goal=goal)


class GoalAdaLNConditioner(nn.Module):
    """AdaLN-Zero goal conditioning (the alternative to cross-attention prefixing).

    Projects the goal vector to an embedding that is **added to the diffusion
    timestep embedding `temb`**, which drives every Cosmos block's AdaLN
    scale/shift/gate. The projection is **zero-initialized**, so at init the added
    embedding is exactly 0 — the timestep conditioning is untouched — and the goal
    then modulates the AdaLN as the projection learns (standard AdaLN-Zero, the DiT
    class-conditioning pattern of adding an extra embedding to the time embedding).

    `temb_dim` is the width of Cosmos's `temb` (= 3 * hidden_size: shift|scale|gate).
    The model wires the output onto `temb` via a forward hook on `time_embed`; this
    module only produces the (B, temb_dim) embedding.
    """

    def __init__(self, goal_dim: int, temb_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, temb_dim),
        )
        # AdaLN-Zero: the final layer is zero-init so the goal contributes nothing at
        # init (timestep conditioning preserved), yet still gets gradient from step 1.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, goal_vec: torch.Tensor) -> torch.Tensor:
        if goal_vec.dim() == 1:
            goal_vec = goal_vec.unsqueeze(0)
        return self.proj(goal_vec)                                       # (B, temb_dim), 0 at init


__all__ = [
    "COSMOS_CROSSATTN_DIM",
    "COSMOS_TEXT_MAX_LEN",
    "DEFAULT_GOAL_TOKENS",
    "ShotProfileCondition",
    "ShotProfileVectorConditioner",
    "GoalAdaLNConditioner",
]
