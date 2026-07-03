# Baseline policies (issue #22)

The comparison set for the WAM (goal-conditioned Cosmos world-action policy).
Every baseline shares the same substrate, so differences come from the policy
class — not the data or metric:

- **Data**: v7 placements (`data/trajectories_full`); trainable baselines use the
  hindsight-relabeled (image, goal, action-chunk) tuples, HER-"future" goal draw.
- **Goal space**: the 8-key V5 shot profile (occupancy, body_in_frame_ratio,
  cam→obj azimuth/elevation, object_center_x/y, bbox_x/y_offset), normalized.
- **Action space**: camera-local 5D `(dx, dy, dz, dyaw, dpitch)` (m / rad, no roll).
- **Val split**: frozen scene manifest `configs/policy/val_scenes.txt` (4 smallest
  scenes, both Forest-fields together) — never trained on, by any baseline.
- **Eval**: shared closed-loop Blender rollout (render → act → move → re-render)
  scored by the geometric goal-profile distance; plus per-family val losses
  during training (loss types differ across families and are NOT comparable —
  use the rollout for head-to-head).

| Baseline | Goal-aware? | Trained? | Backbone / head | What it ablates vs WAM |
|---|---|---|---|---|
| **Diffusion Policy** | yes (FiLM) | yes | frozen DINOv2-L → conditional 1D U-Net, DDPM | no world model, no value head |
| **VLA (π0-style)** | yes (soft tokens) | yes | Qwen3-VL-2B (frozen vision) → flow-matching action expert | no world model, no value head |
| **LLM policy** | yes (language brief) | training-free | VLM API (gemini-2.5-flash via LetSur) prompted per step | no training, implicit previsualization |
| **AutoPhoto** | **no** (aesthetic only) | yes (RL) | aesthetic ResNet18 reward + RecurrentPPO (LSTM) | no goal conditioning at all |
| **UNIC** | **no** (generic composition) | training-free | vendored UNIC composition detector → heuristic pan/zoom | no goal, no learning, reactive only |

## Diffusion Policy (`src/policy/diffusion_policy/`)
Canonical Chi et al. 2023 setup: a **frozen DINOv2-large** encodes the frame to a
pooled embedding; the normalized goal goes through a small MLP; `[obs ‖ goal]`
conditions a **ConditionalUnet1D over the 8-step action chunk via FiLM**
(per-channel scale+bias in every residual block). Trained with DDPM
ε-prediction (100 steps), sampled with DDIM (16). Val = fixed-timestep noise
loss. Train: `scripts/train_diffusion_policy.py` (DDP, loss-only val, early
stopping, resume). *Trained on old data: best val `noise_loss` 0.0050 @ 68k
(50k→early-stop 88k), `runs/old_data/diffusion_policy_best/`.*

## VLA, π0-style (`src/policy/vla/`)
**Qwen3-VL-2B** (vision tower frozen — its per-patch forward is the
throughput wall; `max_pixels` caps input to ~160×128) encodes the frame; the
goal is projected to 4 soft tokens appended to the VLM hidden states; a
**flow-matching ActionExpert** (512-d, 6 layers) cross-attends to that context
and denoises the action chunk (same flow convention as the WAM). Val =
fixed-σ flow loss. Train: `scripts/train_vla_policy.py`. *Trained on old data:
best val `flow_loss` 0.0312 @ 48k/50k, `runs/old_data/vla_best/`.*

## LLM policy (`src/policy/llm_policy/`)
Training-free VLM-as-policy (Photo Agent style). The shot profile is rendered
into a **natural-language framing brief** (`prompt.py`); the model sees the
current frame + brief and must return strict JSON
`{reasoning, dx, dy, dz, dyaw_deg, dpitch_deg}` — one modest move per step.
Responses go through robust parsing (fence/brace extraction, trailing-comma and
quote repair), **field validation, and one corrective re-prompt** before a
no-op fallback; actions clamp to ±3 m / ±60° (`response.py`, `policy.py`).
Backends: local Qwen3-VL-2B (dev) or OpenAI-compatible API — **gemini-2.5-flash
via the LetSur gateway** for real runs (cheap tier by policy; `max_tokens` 2048
because Flash "thinks" before the JSON). No learned parameters.

## AutoPhoto (`src/policy/autophoto/`)
The RL baseline, **goal-agnostic**: reuses AutoPhoto's pretrained **aesthetic
ResNet18** (low-pass, sign-flipped so higher = better) as reward and its 512-d
features as the observation; a fresh **RecurrentPPO (PPO+LSTM)** picks among 9
discrete fine-turn actions (mapped to our 5D deltas) with a terminate action,
inside `PhotoEnv` over a persistent **EEVEE** Blender worker. Faithful
simplifications vs the paper (dense improvement reward, ~1e5 vs 1.5M steps) in
`REFERENCES.md`. Rollouts are render-bound (~4–11 s/step), so training
parallelizes across `train.n_envs` Blender workers (SubprocVecEnv, disjoint
placement shards, one GPU pin per env) — ~8× wall-clock at 8 envs. Train:
`scripts/train_autophoto.py`.

## UNIC (`src/policy/unic/`)
Training-free reactive composition baseline: the vendored **UNIC** model
recommends a well-composed crop box for the current frame; the box's center
offset becomes **pan** (yaw/pitch via the camera FOV) and its width becomes a
monotonic **dolly** heuristic (tight crop → move in, spilling box → back off).
No goal, no previsualization, lateral translation fixed at 0.

## Status / retraining note
The DP and VLA checkpoints above were trained on the **old v7 data** and are
archived under `runs/old_data/` (one best per baseline; val numbers are
per-family losses, not cross-comparable). AutoPhoto needs no trajectory data
(RL in sim, scenes+objects only). When the new dataset lands: retrain DP + VLA
with the same configs, re-run AutoPhoto if scenes changed, and compare all
baselines + WAM with the closed-loop rollout on the held-out scenes.
