# Baseline policies (issue #22)

_Last updated 2026-09-07. Results are on the shared 64-episode manifest
(`configs/eval_episodes_v1.json`) — see Results. This supersedes the pre-pivot version, which
described a hand-rolled Qwen3-VL "π0-style" VLA, a 5D action space, and a 4-scene val split, all
retired (see History)._

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
- **Eval episodes**: **`configs/eval_episodes_v1.json`** — 64 materialised episodes shared with
  V12, so a baseline row and a WAM row are computed on the same shots. `d_start` is a checksum;
  a run that selects its own episodes is not comparable to anything (see the caveat in Results).
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
180k iters (~8 epochs), early-stopped on val at **best 0.0039**. Superseded as the
strongest baseline by pi0.5 — see Results.

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

## Results — shared 64-episode manifest

The ranking number. Episodes come from **`configs/eval_episodes_v1.json`**, the manifest
jungwoo published on `DronePhotographerV12@002-goal-authoring` so a baseline row and a WAM
row can sit in one table: 64 materialised episodes, 8 per view sector, spread over all eight
held-out scenes, `mean d_start = 0.866`.

Run with `scripts/closed_loop_eval_baselines.py --episodes-from configs/eval_episodes_v1.json`.
Protocol matches V12's `closed_loop_eval.py` exactly: distance
`d = sqrt(great_circle(az,el)^2 + dsize^2 + daim^2)` in **radians** (`src/policy/common/reward.py`,
byte-identical to theirs), adaptive horizon `ceil(delta/8)` chunks, chunk-boundary re-render,
shoot-stop at 0.5. `d_start` is a checksum recomputed from the manifest's own poses — the loader
**refuses to run** on a mismatch. Ours passes 0/64 failures (worst 1.4e-08), and our facing map
reproduces their `bearing` on 64/64.

| policy | mean imp | median | frac+ | d_end | d_best | @0.05 | @0.10 | @0.20 | shoot |
|---|---|---|---|---|---|---|---|---|---|
| **pi0.5** @300k | **0.612** | **0.535** | **0.92** | **0.254** | **0.226** | **11%** | **30%** | **56%** | 26/64 |
| pi0.5 @270k | 0.599 | 0.487 | 0.92 | 0.266 | 0.223 | 11% | 31% | 55% | 24/64 |
| **DP** (converged) | 0.357 | 0.294 | 0.72 | 0.508 | 0.425 | 0% | 5% | 17% | 42/64 |
| **GR00T** @130k | 0.235 | 0.238 | 0.64 | 0.631 | 0.479 | 5% | 16% | 28% | 33/64 |

Columns: `imp = d_start - d_end` (a no-op scores exactly 0); `frac+` = fraction of episodes that
improved at all; `d_best` = closest point on the path, so `d_best << d_end` means the policy passed
the shot and drifted off; `@tau` = fraction finishing within tau radians. Summaries in
`runs/shared/*/summary.json`, each carrying the manifest `key` so they join against a WAM run.

**Closure and delivery rank differently, so both belong in the table.** `improvement` says the
camera got closer; it does not say the requested photograph was taken. V12's
`calibrate_shot_threshold.py` finds ~66% of real frame pairs under d<0.10 carry the same six
cinematography words, ~0% past 0.60. DP beats GR00T on every closure column yet GR00T delivers the
shot three times more often (16% vs 5% @0.10) — DP reliably gets *moderately* closer and almost
never lands the shot, GR00T either nails it or misses badly.

**pi0.5 wins both**, and its two checkpoints agree closely (0.612 / 0.599), so this is a property
of the policy rather than a lucky draw. It also stops well: `d_best` 0.226 vs `d_end` 0.254 means
almost no overshoot, and its success@0.10 is identical at best and end — while GR00T loses 19%→16%
by not stopping at its closest point. For reference, V12's own v5 @40k sits at `d_end` 0.35–0.40
with 17–25% inside 0.10.

### Open-loop reconstruction — a fit check, NOT a ranking

| Policy | recon dist | rotation | within 20cm |
|---|---|---|---|
| **pi0.5** @300k | **18.9 cm** | **1.30°** | **80%** |
| GR00T @130k | 31.0 cm | 1.90° | 36% |
| **DP** (converged) | 39.7 cm | 7.00° | 40% |

⚠️ **Reconstruction anti-correlates with closed-loop control.** V12's `heldout_loss.py` notes "v11
found its sampled action MSE was ANTI-correlated with closed-loop success. Treat this as a
fit/overfit tripwire." Do not rank policies by recon cm. Validation curves come from
`scripts/val_sweep.sh` and are written to TensorBoard as `val/*`; they exist to pick a checkpoint
the way DP's early-stopped `ckpt_best` was picked, not to compare policies.

pi0.5's val curve kept descending to the end (190k 22.2 → 260k 20.6 → 300k **18.9** cm), so its
mid-run apparent plateau was noise, not convergence.

### A caveat that invalidated an earlier version of this table

Before the shared manifest, `closed_loop_eval_baselines.py` selected its own episodes with
`select_placements`, which took the first N of a sorted listing. `Chill-Camping` sorts first among
the eight val scenes and holds 60 placements, so **all 8 episodes came from a single room** — a
"held-out" claim that exercised one scene of eight. At n=8, V12's bootstrap puts the 5–95% range for
one policy at [0.21, 0.84], wider than the gaps that table reported. Those numbers ranked nothing.
Always run with `--episodes-from`.

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

## Status (2026-09-07)

- **pi0.5** (`runs/pi05_fair/`) — **done.** 300k steps, 4.34 epochs, final loss 0.015, no
  divergence. Val-best is the final checkpoint (300k, 18.9 cm). This is the strongest baseline.
- **DP** (`runs/20260818_064526_diffusion_policy_dinov2/`) — **done**, early-stopped on val.
- **GR00T** (`runs/groot_fair/`) — **incomplete.** Two failures, in order:
  1. the first fair run diverged at the stock LR (grad-norm creep 0.8 → 6.3 → NaN at 168k; every
     checkpoint past ~130k is NaN, and 150k had already degraded to 320 cm);
  2. the retrain at half LR (5e-5) was stable but **died at step 210k of 300k** on 2026-09-04 when
     rank 1 took a SIGABRT and torchrun tore down rank 0. Cause not yet diagnosed; the 210k
     checkpoint is intact and predicts finite actions.

  The published GR00T row is therefore **@130k**, i.e. budget-unequal against pi0.5's 300k and in
  GR00T's disfavour. A 210k evaluation is in flight; if it moves the numbers, this table gets the
  210k row and the caveat is dropped.
- **Not yet re-run on the new substrate:** LLM policy, AutoPhoto, UNIC.

A note on stopping runs early: GR00T was first halted at 130k because "its val curve had
flattened", but its last two points were still descending (34.3 → 31.0 cm), and the identical read
on pi0.5 at 190k proved wrong — it went on to 18.9. Prefer the full budget, or a proper
early-stopping rule, over eyeballing a noisy curve.

TensorBoard: one canonical tag scheme across every run (`train/loss`, `train/lr`, `train/grad_norm`,
`val/*`) via `scripts/tb_align.py`; the VLAs get TB from `scripts/tb_from_log.py`, since
lerobot-train writes none and wandb is off. `tensorboard --logdir runs`.

## Reproduce
- Export data: `scripts/export_lerobot.py` → `lerobot_pi05_v3` (the padding-free export).
- Train: `scripts/train_pi05.sh`, `scripts/train_groot.sh` (see per-run env vars).
- Recon: `scripts/check_reconstruction_{dp,lerobot}.py --scenes val`.
- Sim (the ranking number): `scripts/closed_loop_eval_baselines.py --policy {dp,pi05,groot}
  --checkpoint <ckpt> --episodes-from configs/eval_episodes_v1.json --out runs/shared/<name>`.
  Without `--episodes-from` it picks its own episodes and the result is not comparable to anything.
- Val curve over all checkpoints: `scripts/val_sweep.sh <run> <policy> <env> "<gpus>"`.
- GIFs: `scripts/make_rollout_gif.py`.
