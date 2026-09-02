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
from src.policy.vla.action_expert import ActionExpert, DiTActionExpert


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
        expert_type: str = "mlp",           # "mlp" (π0-style) | "dit" (GR00T-style AdaLN-Zero)
        freeze_backbone: bool = False,
        freeze_vision: bool = False,
        flow_config: FlowConfig | None = None,
        action_scale=None,
        processor=None,
        prompt: str = "Describe the camera framing of the subject.",
        goal_conditioning: str = "soft_token",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.processor = processor          # Qwen3VLProcessor (real path); None for mock tests
        self.prompt = prompt
        # "soft_token": goal -> soft tokens appended to context (fixed text prompt).
        # "text": the goal IS the prompt (jungwoo's goal_prompt, subject-relative bearing);
        #         no goal tokens, goal_vec is ignored. See src/policy/common/goal_text.py.
        self.goal_conditioning = goal_conditioning
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
        # Only built for soft-token conditioning: in "text" mode the goal is in the
        # prompt and these would be unused params (a hard DDP error under all-reduce).
        if self.goal_conditioning != "text":
            self.goal_proj = nn.Linear(goal_dim, n_goal_tokens * ctx_dim)
            self.goal_norm = nn.LayerNorm(ctx_dim)
        # "mlp" = the π0-style add-timestep expert; "dit" = GR00T-style AdaLN-Zero DiT head.
        _Expert = DiTActionExpert if expert_type == "dit" else ActionExpert
        self.action_expert = _Expert(
            action_dim=action_dim, ctx_dim=ctx_dim, dim=expert_dim,
            depth=expert_depth, n_heads=expert_heads, chunk_size=chunk_size,
        )

        import numpy as np

        scale = ACTION_SCALE if action_scale is None else action_scale
        self.register_buffer("action_scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif freeze_vision:
            # Freeze only the vision tower; finetune the LM + expert + goal proj.
            # The vision tower is the slow part (its forward still runs every step),
            # so this is a baseline-design + memory choice, not a big speedup.
            vis = getattr(self.backbone, "visual", None) or getattr(self.backbone, "vision_tower", None)
            if vis is None:
                raise AttributeError("could not find the vision tower on the backbone (.visual / .vision_tower)")
            n = 0
            for p in vis.parameters():
                p.requires_grad_(False); n += 1
            print(f"[VLA] froze vision tower: {n} param tensors", flush=True)

    def prepare_inputs(self, batch: dict, device, dtype) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Build (vlm_inputs, goal_vec, action_chunk) from a dataloader batch.

        Runs the Qwen3-VL processor on the [-1,1] CHW images (a fixed text prompt
        per sample; the goal enters separately as soft tokens, not as text). The
        mock tests bypass this and call `compute_loss` with hand-built tensors.
        """
        from src.policy.vla.dataset import build_vlm_inputs, goal_to_prompt

        if self.goal_conditioning == "text" and "goal_raw" in batch:
            objs = [m["object"] for m in batch["meta"]] if "meta" in batch else batch["object"]
            prompt = [goal_to_prompt(gr.cpu().numpy(), o) for gr, o in zip(batch["goal_raw"], objs)]
        else:
            prompt = self.prompt
        proc = build_vlm_inputs(self.processor, prompt, batch["state_image"])
        vlm_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
        goal = batch["goal_vec"].to(device, dtype)
        action = batch["action_chunk"].to(device, dtype)
        return vlm_inputs, goal, action

    def forward_context(self, vlm_inputs: dict, goal_vec: torch.Tensor) -> torch.Tensor:
        """Run the VLM; append goal tokens unless the goal is already in the text prompt."""
        out = self.backbone(**vlm_inputs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.goal_conditioning == "text":
            return h                        # goal is in the prompt; no soft tokens
        b = h.shape[0]
        goal_tok = self.goal_proj(goal_vec).view(b, self.n_goal_tokens, self.ctx_dim)
        goal_tok = self.goal_norm(goal_tok).to(h.dtype)
        return torch.cat([h, goal_tok], dim=1)

    def compute_loss(self, vlm_inputs: dict, goal_vec: torch.Tensor, action_chunk: torch.Tensor,
                     sigma: Optional[torch.Tensor] = None) -> VLALossOutputs:
        """Flow-matching velocity MSE on the action chunk (a0 = ground-truth chunk).

        `sigma` (B,) overrides the random draw — validation passes a fixed grid so
        the metric is comparable across checkpoints.
        """
        ctx = self.forward_context(vlm_inputs, goal_vec)
        a0 = action_chunk
        b = a0.shape[0]
        if sigma is None:
            sigma = sample_flow_sigma(b, self.flow, device=a0.device)
        sigma = sigma.to(a0.dtype)
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
            scale = self.action_scale.to(x.dtype)   # (POSE_DIM,)
            p = scale.shape[-1]
            if x.shape[-1] > p:
                # 6D action (pose + shoot): scale pose dims, pass shoot (0/1) through.
                x = torch.cat([x[..., :p] * scale, x[..., p:]], dim=-1)
            else:
                x = x * scale
        return VLAOutputs(pred_action_chunk=x)


__all__ = ["VLAActionPolicy", "VLALossOutputs", "VLAOutputs"]
