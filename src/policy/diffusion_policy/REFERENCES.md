# Diffusion Policy baseline — references & attribution

Diffusion Policy ablation baseline (issue #22). Not a vendored codebase — a
from-scratch reimplementation against our infra, informed by:

| Source | What we took |
|---|---|
| **Diffusion Policy** (Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023 — diffusion-policy.cs.columbia.edu, github.com/real-stanford/diffusion_policy, MIT) | The core method: a conditional **DDPM over an action chunk**, with observations as **global (FiLM) conditioning**. Our `denoiser.py` reimplements its recommended CNN backbone `ConditionalUnet1D` (1D U-Net over the temporal axis, sinusoidal step embedding, FiLM residual blocks). |
| **DINOv2** (Oquab et al., 2023; `facebook/dinov2-large`, Apache-2.0) | The "latest vision backbone" — used **frozen** as the observation encoder (global `pooler_output`, 1024-d). |
| **diffusers** (`DDPMScheduler` / `DDIMScheduler`, Apache-2.0) | The noise schedule: DDPM (epsilon-prediction) for training, DDIM for fast sampling at inference. |

## Design choices (confirmed)
- **Frozen DINOv2-large** encoder + trained `ConditionalUnet1D` head — the canonical
  Diffusion Policy regime and the point of a strong pretrained backbone. Frozen ⇒
  large batch fits easily, so the head trains fast (no bs=1 kernel-launch stall).
- **DDPM/DDIM** action head (not flow matching) — keeps this a faithful, recognizable
  *Diffusion Policy*; varying the generative paradigm from our flow head is a
  feature, since the **world model** stays the single ablated variable.
- **Goal as a vector** (embedded, concatenated to the obs embedding) — same goal
  *information* as ours, not text.

## What makes it an *ablation* of our method
Identical to the Cosmos world-action policy except the world model and the
backbone family: same v7 data + windows + HER goal sampling (`common/dataset_base`),
same goal space + normalization (`common/goal_space`), same 5D action + `ACTION_SCALE`
(`common/action_repr`), same held-out `val_scenes.txt`, same eval metric
(`common/reward.score_distance`). Removed: future-frame prediction (the world
model) and the value head; the VLM is replaced by a frozen DINOv2 encoder. A win
for Cosmos over this baseline is attributable to previsualization.
