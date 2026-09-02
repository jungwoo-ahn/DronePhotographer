# Baseline policies (issue #22)

_Last updated 2026-09-02. This supersedes the pre-pivot version, which described a
hand-rolled Qwen3-VL "π0-style" VLA, a 5D action space, and a 4-scene val split —
all retired (see history below)._

The comparison set for the WAM (goal-conditioned Cosmos world-action policy).
Every baseline shares the same substrate, so differences come from the policy
class — not the data or metric:

- **Data**: v7 placements (`data/trajectories_full`); trainable baselines use the
  hindsight-relabeled (image, goal, action-chunk) tuples via the **goal-start
  sampler** (8-action windows: start → 8-step walk → achieved goal). The
  pretrained VLAs train on the LeRobot export `lerobot_pi05_v3` (138k episodes,
  1.1M frames), built by `scripts/export_lerobot.py`.
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

_Retired: the hand-rolled **Qwen3-VL "π0-style" VLA** (now `legacy/src/policy/vla/`) was replaced by
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

All numbers below are on the **clean `lerobot_pi05_v3` export** and the fair
retrains. The earlier 100k/`v2` numbers are superseded — see History.

### Closed-loop — the metric that decides the ranking

Ported from `DronePhotographerV12/scripts/closed_loop_eval.py` so these are directly
comparable to the WAM (`scripts/closed_loop_eval_baselines.py`). Distance is
`d = sqrt(great_circle(az,el)^2 + dsize^2 + daim^2)` in **radians**
(`src/policy/common/reward.py`, byte-identical to V12's), headline
`improvement = d_start - d_end` (a no-op scores exactly 0). Adaptive horizon
`ceil(|goal_idx - start_idx| / 8)` chunks, chunk-boundary re-render, shoot-stop at 0.5.

**val**, n=8, same episodes for all three (`mean d_start` 0.721):

| Policy | mean imp | median | frac+ | best imp | d_end | % gap closed |
|---|---|---|---|---|---|---|
| **pi0.5** @190k | **0.455** | 0.358 | 0.62 | **0.538** | **0.266** | 43% |
| **DP** (converged) | 0.409 | **0.433** | **0.75** | 0.469 | 0.312 | 43% |
| **GR00T** @130k | 0.284 | 0.147 | 0.50 | 0.450 | 0.437 | 14% |

**train** (V12's `--split train` control), n=6, `mean d_start` 0.861: pi0.5 0.613 /
DP 0.461 / GR00T 0.305. By % gap closed: DP 38%→43% (no generalization gap), pi0.5
53%→43% (modest), GR00T 14%→14% (uniformly weak, not overfit).

Raw per-episode traces: `outputs/v12metric_results/` (gitignored, this box only).

### Open-loop reconstruction — useful for fit, NOT for ranking

| Policy | recon dist | rotation | within 20cm |
|---|---|---|---|
| **pi0.5** @190k | **22.2 cm** | 1.60° | **72%** |
| **GR00T** @130k | 31.0 cm | **1.90°** | 36% |
| **DP** (converged) | 39.7 cm | 7.00° | 40% |

⚠️ **Reconstruction anti-correlates with closed-loop control.** Recon ranks pi0.5
~1.8x above DP; closed-loop they tie, with DP *better* on median and consistency.
V12 hit the same thing independently — `src/train/heldout_loss.py` there notes "v11
found its sampled action MSE was ANTI-correlated with closed-loop success. Treat this
as a fit/overfit tripwire." **Do not rank policies by recon cm.**

Validation curves (`scripts/val_sweep.sh`, written to TensorBoard as `val/*`) pick each
VLA's best checkpoint the way DP's `ckpt_best` was picked. pi0.5 was still improving at
190k (30.9 → 22.2 cm over 50k→190k); GR00T flattened (43.4 → 31.0 over 50k→130k) and was
stopped there.

## History — two bugs that invalidated the first round

Kept because both were silent and either could recur.

1. **Zero-action padding poisoned the export.** Each episode's final, actionless frame
   was written with a zero 10D action (~11% of frames). Those rows dragged the rot6d
   quantiles from [0.994, 1.0] to [0, 1], so under pi0.5's QUANTILES normalization the
   real rotation signal occupied <1% of the output range. Fixed in
   `scripts/export_lerobot.py` (emit only frames with a real outgoing action) → `v3`.
   Effect at 10k steps: pi0.5 rotation **57.3° → 4.4°**, GR00T **255 cm → 85 cm**. The
   zeros were also poison *targets* ("from here, do nothing"), which is why translation
   improved too.
2. **Unfair budget + an LR floor.** The first VLA runs did 0.59 epoch vs DP's ~8, and
   pi0.5's openpi preset (`scheduler_decay_steps` default 30000) floored the LR at step
   30k of a 100k run — ~70% of training at min-LR. Fixed by passing
   `--policy.scheduler_decay_steps=<STEPS>`.

Also: GR00T's first fair run **diverged** (grad-norm creep → NaN at 168k; every
checkpoint after ~130k is NaN, and 150k had already degraded to 320 cm). Retrained at
half LR (5e-5 vs the 1e-4 default), which was stable. Its 130k checkpoint is final.

## Status (2026-09-02)
- **pi0.5** (`runs/pi05_fair/`): training, ~200k/300k, DDP eff-batch 16 on `v3`.
  Best checkpoint so far 190k.
- **GR00T** (`runs/groot_fair/`): **final at 130k** — stopped once its val curve
  flattened. Budget is therefore unequal vs pi0.5; state that when quoting the gap.
- **Not yet re-run on the new substrate:** LLM policy, AutoPhoto, UNIC.

TensorBoard: one canonical tag scheme across every run (`train/loss`, `train/lr`,
`train/grad_norm`, `val/*`) via `scripts/tb_align.py`; the VLAs get TB from
`scripts/tb_from_log.py` (lerobot-train writes none and wandb is off). `tensorboard --logdir runs`.

## Reproduce
- Export data: `scripts/export_lerobot.py` → `lerobot_pi05_v3` (the padding-free export).
- Train: `scripts/train_pi05.sh`, `scripts/train_groot.sh` (see per-run env vars).
- Recon: `scripts/check_reconstruction_{dp,lerobot}.py --scenes val`.
- Sim (V12 metric, the ranking number): `scripts/closed_loop_eval_baselines.py --policy {dp,pi05,groot} --split {val,train}`.
- Val curve over all checkpoints: `scripts/val_sweep.sh <run> <policy> <env> "<gpus>"`.
- GIFs: `scripts/make_rollout_gif.py`.
