# Baseline policies (issue #22)

_Last updated 2026-08-27. This supersedes the pre-pivot version, which described a
hand-rolled Qwen3-VL "π0-style" VLA, a 5D action space, and a 4-scene val split —
all retired (see history below)._

The comparison set for the WAM (goal-conditioned Cosmos world-action policy).
Every baseline shares the same substrate, so differences come from the policy
class — not the data or metric:

- **Data**: v7 placements (`data/trajectories_full`); trainable baselines use the
  hindsight-relabeled (image, goal, action-chunk) tuples via the **goal-start
  sampler** (8-action windows: start → 8-step walk → achieved goal). The
  pretrained VLAs train on the LeRobot export `lerobot_pi05_v2` (150k episodes,
  1.35M frames), built by `scripts/export_lerobot.py`.
- **Goal space**: the shot profile, with **`subject_bearing_deg`** replacing
  `cam_to_obj_azimuth_deg` (subject-frame bearing via `facing_map_final.json`).
  DP receives the goal as a **numeric vector**; the pretrained VLAs receive it as
  a **natural-language prompt** (`goal_text.goal_prompt`) — a design difference to
  keep in mind when comparing.
- **Action space**: **10D** `[Δtranslation_cam (3), rot6d (6), shoot (1)]`, raw /
  unnormalized (`POSE_DIM=9`, `ACTION_DIM=10`; world-azimuth yaw per V12). The
  `shoot` channel is a learned "declare arrival" bit.
- **Val split**: frozen scene manifest **`configs/policy/val_scenes.json`** — the
  **8 held-out object-disjoint scenes** (jungwoo's 009 split); never trained on,
  by any baseline (leakage audited: DP config carries `val_split_level=scene`, and
  the VLA export writes train/val as separate datasets).
- **Eval**: (1) **open-loop reconstruction** — sample the 8-action chunk, apply it
  from the start pose, measure cm / deg to the walk endpoint (`check_reconstruction_dp.py`,
  `check_reconstruction_lerobot.py`; GT-chunk sanity must be ~0); (2) **closed-loop
  Blender rollout** (render → act → move → re-render) scored by goal-profile
  distance (`rollout_dp.py`, `rollout_vla.py`, shared `rollout_server.py`).
  **Per-family training losses are NOT comparable** (DDPM ε-MSE vs flow-matching
  velocity-MSE, different normalization / timestep weighting / dim-averaging) —
  use recon + rollout for head-to-head.

| Baseline | Goal-aware? | Trained? | Backbone / head | What it ablates vs WAM |
|---|---|---|---|---|
| **Diffusion Policy** | yes (FiLM, vector goal) | yes | frozen DINOv2-L → conditional 1D U-Net, DDPM | no world model, no value head |
| **pi0.5** (real pretrained VLA) | yes (NL goal) | fine-tuned | LeRobot `pi05_base` ~3B (PaliGemma + Gemma action expert), flow-matching | pretrained VLA prior, no world model |
| **GR00T N1.7** (real pretrained VLA) | yes (NL goal) | fine-tuned | LeRobot `groot` (Cosmos-Reason2-2B backbone + flow action head) | pretrained VLA prior, no world model |
| **LLM policy** | yes (language brief) | training-free | VLM API (gemini-2.5-flash via LetSur) prompted per step | no training, implicit previsualization |
| **AutoPhoto** | **no** (aesthetic only) | yes (RL) | aesthetic ResNet18 reward + RecurrentPPO (LSTM) | no goal conditioning at all |
| **UNIC** | **no** (generic composition) | training-free | vendored UNIC composition detector → heuristic pan/zoom | no goal, no learning, reactive only |

_Retired: the hand-rolled **Qwen3-VL "π0-style" VLA** (`src/policy/vla/`) was replaced by
the two real pretrained VLAs above (pi0.5, GR00T) loaded via LeRobot — see `pi05-baseline-setup`
/ `groot-baseline-setup` for env + vendoring details._

## Diffusion Policy (`src/policy/diffusion_policy/`)
Canonical Chi et al. 2023 setup: a **frozen DINOv2-large** encodes the frame to a
pooled embedding; the normalized **goal vector** goes through a small MLP; `[obs ‖ goal]`
conditions a **ConditionalUnet1D over the 8-step action chunk via FiLM**
(per-channel scale+bias in every residual block). Trained with DDPM
ε-prediction (100 steps), sampled with DDIM (16). Train:
`scripts/train_diffusion_policy.py` (DDP, loss-only val, early stopping, resume).
**Current run**: `runs/20260818_064526_diffusion_policy_dinov2/` — batch 64 ×
180k iters (~8 epochs), early-stopped on val at **best 0.0039**. This is the
strongest baseline to date.

## pi0.5 (LeRobot, real pretrained VLA)
`lerobot/pi05_base` (~3B: PaliGemma vision-language + Gemma action expert),
**fine-tuned full** (no freeze) on our data. Frame + zero proprio state + the
**NL goal prompt**; a **flow-matching** action expert denoises the 10D chunk.
Action norm **QUANTILES**, VISUAL/STATE `IDENTITY` (state is all-zeros — no
proprio). Env `vla` (lerobot 0.6.2, transformers 5, py3.12); PaliGemma tokenizer
vendored + run offline. Train: `scripts/train_pi05.sh` (`PARALLEL=ddp NPROC=2`
for multi-GPU DDP; **must** pass `--policy.scheduler_decay_steps=<STEPS>` so
cosine spans the full run — see fairness note). Details: `pi05-baseline-setup`.

## GR00T N1.7 (LeRobot, real pretrained VLA)
`nvidia/GR00T-N1.7-3B` via LeRobot `groot` (backbone = **Cosmos-Reason2-2B**,
Qwen3-VL-based, bundled; + diffusion/flow action head). Same substrate: frame +
**state** (GR00T needs `observation.state`, unlike pi0.5) + NL goal prompt →
10D chunk. Its cosine scheduler already spans `--steps` (no LR-floor fix needed).
Env `vla_groot` (clone of `vla` + peft/diffusers/timm/dm-tree/decord). Train:
`scripts/train_groot.sh` (`PARALLEL=ddp NPROC=2`). Details: `groot-baseline-setup`.

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

## Results (8 held-out val scenes)

**Open-loop reconstruction** (sampled 8-action chunk → walk endpoint; GT-chunk
sanity 0.000 cm, so the metric is valid):

| Policy | recon dist | rotation | within 20cm | fails on |
|---|---|---|---|---|
| **DP** (dinov2) | **39.7 cm** | **7.0°** | **40%** | — (best on both) |
| **pi0.5** (100k) | 75 cm | 57.3° | 15% | rotation (rot6d r0/r4 MSE≈0.35) |
| **GR00T** (100k) | 258–278 cm | 17° | 0% | translation / depth (tz MSE≈0.28) |

**Closed-loop sim** (own-goal, non-trivial gaps): DP reaches 2/3 (Nature-Snowy
0.63→0.085, basement 0.26→0.105) and halves Parking; pi0.5 halves the far goals
but is erratic; GR00T makes minimal progress (emits large ~constant moves largely
**independent of goal distance** — moves ~1 m even when already at goal). GIFs +
viewer: `outputs/dp_rollout_gifs/` (published artifact). Goal-dependence probe
(`check_reconstruction_lerobot.py --shuffle_goals`) queued to confirm whether
GR00T ignores the goal.

The two VLAs fail in **opposite dimensions** (pi0.5 rotation, GR00T translation) —
so the failure is backbone/action-head specific, not the NL-goal encoding (pi0.5
gets translation right from the same prompt).

## ⚠️ Fairness caveat — the 100k VLA numbers above are UNFAIR
Both 100k VLAs trained only **0.59 epoch** (batch 8 × 100k on 1.35M frames),
while DP trained **~8 epochs** and early-stopped at convergence. Worse, **pi0.5
had an LR-floor bug**: its openpi cosine preset (`scheduler_decay_steps=30000`
default) floored the LR at step 30k while the run went to 100k → ~70% of training
at min-LR. So DP's lead is **provisional**; do not cite the raw 100k VLA numbers
as an architecture verdict.

## Status (2026-08-27)
**Fair retrains in progress.** DDP via `PARALLEL=ddp NPROC=2` (plain torchrun
auto-fills `dp_replicate=world_size`; the model fits one 49 GB GPU at per-GPU
batch 8, so this is data-parallel, not FSDP — FSDP had a DTensor bug).
- **pi0.5_fair** (`runs/pi05_fair/`): GPUs 6+7, eff batch 16, **300k steps ≈ 3.5
  epochs**, cosine over the full 300k (LR-fix). In progress (~1/3 done, loss
  0.11 and dropping).
- **GR00T_fair** (`runs/groot_fair/`): queued — launches on pi0.5's completion
  (2 GPUs needed for DDP), same ~3.5-epoch budget.
- Old 100k runs kept at `runs/{pi05,groot}_drone/checkpoints/100000` as the
  documented (unfair) baseline of record.

TensorBoard: DP has native TB (`runs/…_dinov2/tb`); the VLAs get TB via
`scripts/tb_from_log.py` (parses train stdout → tfevents; wandb is off). View all
with `tensorboard --logdir runs`.

**Not yet re-run on the new substrate:** LLM policy, AutoPhoto (`runs/autophoto/`
is an older run), UNIC. Re-run + fold into the results table before the final WAM
comparison.

## Reproduce
- Export data: `scripts/export_lerobot.py` → `lerobot_pi05_v2`.
- Train: `scripts/train_pi05.sh`, `scripts/train_groot.sh` (see per-run env vars).
- Recon: `scripts/check_reconstruction_{dp,lerobot}.py --scenes val`.
- Sim: `scripts/rollout_{dp,vla}.py --own_goal`; batch via `scripts/vla_val_eval.sh`.
- GIFs: `scripts/make_rollout_gif.py`.
