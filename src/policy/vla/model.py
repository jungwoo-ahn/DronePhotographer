"""VLAActionPolicy — π0-style VLA ablation baseline ("ours without the WAM").

A VLM (Qwen3-VL-2B) encodes the current image; the normalized shot-profile goal
is projected to soft tokens and concatenated to the VLM hidden states; a
flow-matching ActionExpert cross-attends to that context and denoises the 5D
action chunk. No future-frame prediction (that is exactly the world-action model
we ablate), no value head.

This isolates the previsualization contribution: same data / goal / action /
flow convention / eval as the Cosmos world-action policy, minus the world model.

Backbone seam: the policy calls `self.backbone(**vlm_inputs).last_hidden_state`.
The real backbone is `Qwen3VLModel`; tests pass a tiny mock with the same
contract, so the whole data → context → flow path runs without the 2B weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.policy.common.action_repr import ACTION_DIM, ACTION_SCALE
from src.policy.common.flow import FlowConfig, flow_sigma_schedule, sample_flow_sigma
from src.policy.vla.action_expert import ActionExpert


@dataclass
class VLALossOutputs:
    """Mirrors cosmos LossOutputs so the trainer logging is shared. Action-only."""

    total: torch.Tensor
    action: torch.Tensor

    def detach_dict(self) -> dict:
        return {"total": float(self.total.detach()), "action": float(self.action.detach())}


@dataclass
class VLAOutputs:
    pred_action_chunk: torch.Tensor    # (B, chunk, ACTION_DIM)


class VLAActionPolicy(nn.Module):
    """π0-style VLA: VLM context + flow-matching action expert.

    Args:
      backbone: a `Qwen3VLModel` (or mock) returning `.last_hidden_state` (B,L,D).
      ctx_dim: backbone hidden size D (read from backbone.config if None).
      goal_dim: normalized goal vector dim (default 8).
      n_goal_tokens: number of soft goal tokens appended to the VLM context.
      action_dim / chunk_size: 5D action over `chunk_size` future steps.
      expert_dim / expert_depth / expert_heads: ActionExpert size.
      freeze_backbone: if True, only the goal proj + action expert train.
      flow_config: shared flow-matching sigma config (defaults to FlowConfig()).
      action_scale: per-dim action normalization (persisted as a buffer).
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        ctx_dim: Optional[int] = None,
        goal_dim: int = 8,
        n_goal_tokens: int = 4,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 8,
        expert_dim: int = 512,
        expert_depth: int = 6,
        expert_heads: int = 8,
        freeze_backbone: bool = False,
        flow_config: FlowConfig | None = None,
        action_scale=None,
        processor=None,
        prompt: str = "Describe the camera framing of the subject.",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.processor = processor          # Qwen3VLProcessor (real path); None for mock tests
        self.prompt = prompt
        if ctx_dim is None:
            cfg = getattr(backbone, "config", None)
            tcfg = getattr(cfg, "text_config", cfg)
            ctx_dim = int(getattr(tcfg, "hidden_size"))
        self.ctx_dim = ctx_dim
        self.goal_dim = goal_dim
        self.n_goal_tokens = n_goal_tokens
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flow = flow_config or FlowConfig()

        # Goal vector -> soft tokens in the VLM hidden space (appended to context).
        self.goal_proj = nn.Linear(goal_dim, n_goal_tokens * ctx_dim)
        self.goal_norm = nn.LayerNorm(ctx_dim)
        self.action_expert = ActionExpert(
            action_dim=action_dim, ctx_dim=ctx_dim, dim=expert_dim,
            depth=expert_depth, n_heads=expert_heads, chunk_size=chunk_size,
        )

        import numpy as np

        scale = ACTION_SCALE if action_scale is None else action_scale
        self.register_buffer("action_scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def prepare_inputs(self, batch: dict, device, dtype) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Build (vlm_inputs, goal_vec, action_chunk) from a dataloader batch.

        Runs the Qwen3-VL processor on the [-1,1] CHW images (a fixed text prompt
        per sample; the goal enters separately as soft tokens, not as text). The
        mock tests bypass this and call `compute_loss` with hand-built tensors.
        """
        from PIL import Image
        import numpy as np

        imgs = batch["state_image"]                       # (B, 3, H, W) in [-1, 1]
        pil = []
        for im in imgs:
            arr = ((im.float().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            pil.append(Image.fromarray(arr))
        b = len(pil)
        messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]] * b
        text = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        proc = self.processor(text=text, images=pil, return_tensors="pt", padding=True)
        vlm_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
        goal = batch["goal_vec"].to(device, dtype)
        action = batch["action_chunk"].to(device, dtype)
        return vlm_inputs, goal, action

    def forward_context(self, vlm_inputs: dict, goal_vec: torch.Tensor) -> torch.Tensor:
        """Run the VLM and append goal tokens → context (B, L + n_goal_tokens, D)."""
        out = self.backbone(**vlm_inputs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        b = h.shape[0]
        goal_tok = self.goal_proj(goal_vec).view(b, self.n_goal_tokens, self.ctx_dim)
        goal_tok = self.goal_norm(goal_tok).to(h.dtype)
        return torch.cat([h, goal_tok], dim=1)

    def compute_loss(self, vlm_inputs: dict, goal_vec: torch.Tensor, action_chunk: torch.Tensor) -> VLALossOutputs:
        """Flow-matching velocity MSE on the action chunk (a0 = ground-truth chunk)."""
        ctx = self.forward_context(vlm_inputs, goal_vec)
        a0 = action_chunk
        b = a0.shape[0]
        sigma = sample_flow_sigma(b, self.flow, device=a0.device).to(a0.dtype)
        eps = torch.randn_like(a0)
        s = sigma[:, None, None]
        a_sigma = (1.0 - s) * a0 + s * eps
        v_pred = self.action_expert(a_sigma, sigma, ctx)
        loss = (v_pred - (eps - a0)).pow(2).mean()
        return VLALossOutputs(total=loss, action=loss)

    @torch.no_grad()
    def sample(self, vlm_inputs: dict, goal_vec: torch.Tensor, *, n_steps: int = 10, denormalize: bool = True) -> VLAOutputs:
        """Euler-integrate the flow sigma:1→0 to produce the action chunk.

        The VLM context is computed once and reused across integration steps.
        `denormalize=True` returns physical units via the action_scale buffer.
        """
        ctx = self.forward_context(vlm_inputs, goal_vec)
        b = ctx.shape[0]
        x = torch.randn(b, self.chunk_size, self.action_dim, device=ctx.device, dtype=ctx.dtype)
        sigmas = flow_sigma_schedule(n_steps, device=ctx.device)
        for i in range(n_steps):
            sig = sigmas[i].to(x.dtype).expand(b)
            v = self.action_expert(x, sig, ctx)
            x = x + (sigmas[i + 1] - sigmas[i]).to(x.dtype) * v
        if denormalize:
            x = x * self.action_scale.to(x.dtype)
        return VLAOutputs(pred_action_chunk=x)


__all__ = ["VLAActionPolicy", "VLALossOutputs", "VLAOutputs"]
