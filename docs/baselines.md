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
| **GPT-5 agent** | yes (brief, goal frame, or measured profile) | training-free | gpt-5 via LetSur, one 5D move per observation | no training, frontier VLM instead of a policy |
| **PhotoFlow-style** | yes (brief, via the Reviewer's `m6`) | training-free | Director/Reviewer/Reflector search over absolute camera states | policy replaced by inference-time search |

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

Pan uses the **real** render camera: 24 mm on a 12.8×9.6 mm sensor at 1024×768, so
hFOV **29.86°** / vFOV **22.62°** (`RENDER_HFOV_DEG`, derived from the constants in
`scripts/rollout_server.py` rather than hardcoded). The previous 50° default
over-rotated every pan by ~1.7×.

## GPT-5 agent (`--policy gpt5`)
Frontier VLM as a camera operator, training-free. It sees the current frame plus a goal and
returns strict JSON `{reasoning, dx, dy, dz, dyaw_deg, dpitch_deg, done}` — one camera-local 5D
move per observation, so it gets `n_chunks` decisions, the same number of renders the learned
policies see at chunk boundaries (**observation-matched**, not step-matched: matching their 8
pose-steps per chunk would hand it 8× the decisions and 8× the API spend for the same evidence).
Three conditioning modes, reported as separate rows because they do not see the same thing:

- `--goal-mode text` — the NL shot brief, exactly what the VLAs are conditioned on. **This is the
  apples-to-apples row.**
- `--goal-mode image` — the goal pose rendered live. Strictly more information (it reveals the
  scene, not just the requested composition).
- `--feedback 1` — the brief *plus* its own measured profile and the signed remaining error
  (bearing wrapped circularly), i.e. PhotoFlow's Reviewer signal. Strictly more again.

`max_tokens` is 8192 with a 16384 retry and **not** taken from `llm_policy_api.yaml`: that file's
2048 was sized for gemini-2.5-flash, and gpt-5 spends ~2.1k *reasoning* tokens on an image prompt,
so at 2048 it returns `finish_reason="length"` with empty content — silently zero-action. 156 calls
per 64-episode run.

## PhotoFlow-style search (`--policy photoflow`, `src/policy/photoflow/`)
**Reimplemented from the paper** (arXiv:2605.23771) — the authors' repo ships only a project page
and VPhotoBench, with Agent Code and Evaluation unreleased, so the prompts and JSON schemas are
ours while every stated constant is theirs: 6 rounds × 4 candidates (24 previews), `J = 0.10·m1 +
0.10·m2 + 0.15·m3 + 0.15·m4 + 0.25·m5 + 0.25·m6`, `m2 = max(0, 1 − d/0.45)` on screen-space
distance, region cells of side `max(0.12·sceneScale, 0.9)`, promising at `.68`/`.70`, and the
relocation score `s(a) = π(a) + u(a) + min(‖p_a − p_t‖/(2h), 2.0) − 0.35·n(a) − 0.40·1[promising]`.
`tests/policy/test_photoflow.py` pins all of them. Label any row from it **"PhotoFlow-style"**, not
as the authors' own result.

Three things make it unlike every other row:
- **It is a global search, not a policy.** It proposes absolute camera states anywhere in the scene
  rather than stepping from the start pose, so `improvement` stops measuring navigation.
- **It gets 24 renders** against the policies' 2–4. `--render-budget` caps it for a render-matched
  ablation.
- **Scope**: the search is over `(position, look_at)`. The paper also searches focal length,
  aperture and aspect ratio; our renderer is fixed at 24 mm / 4:3 and the metric cannot see them,
  so varying them would be unmeasurable rather than faithful. `m1`/`m2` are computed from the
  *brief's* requested half and composition point — feeding the continuous goal-distance into them
  would hand the search its own objective, since `d` is built from bearing, elevation and size.

The Director's scene scout (object names, bboxes, centres, scene extents, camera intrinsics) is
written once per placement by `scripts/rollout_server.py` at load.

## The 5D action conventions (read this before adding a baseline)

Every non-learned baseline here emits the legacy 5D action `(dx, dy, dz, dyaw, dpitch)`.
Two of its properties are not self-evident from the code and were wrong in three places
at once, so they are now pinned by `tests/policy/test_action_5d_conventions.py`:

- **`dyaw` is LEFT-positive.** `apply_action_5d` yaws about world up, and with
  `right = cross(forward, WORLD_UP)` a positive `dyaw` swings the forward vector *away*
  from the camera's right axis. Human-facing descriptions ("pan right") are
  right-positive, so anything that speaks in those terms must convert through
  `yaw_from_right_positive()`. The GPT-5 agent prompt, UNIC's pan mapping and
  `src/policy/llm_policy/` all assumed the signs agreed; every yaw was inverted.
- **`POSE_DIM_5D` is 5, `POSE_DIM`/`ACTION_DIM` are 9/10.** The V12 port moved the
  latter from 5/6 to 9/10. Code that still said `np.zeros(ACTION_DIM)` silently grew
  four unused rot6d slots, and the 5D guards — which read `POSE_DIM`/`ACTION_DIM` —
  began *rejecting* 5D input while *accepting* 9D, reading `rot6d[0:2]` as
  `(dyaw, dpitch)` radians. That fixed 7 pre-existing test failures.

Also mind the **lens**: a 29.9° horizontal frame at a median 4.95 m subject distance means
a ~1.3 m uncompensated sideways move puts the subject at the frame edge, and there is no
recovery once it is gone. A prompted agent that is not *told* this will wander out of frame
and then search blind — worth −1.56 of mean improvement on its own (see below).

Per-decision clamps are matched to what the learned policies actually do in one 8-step chunk,
measured over 867 chunks of `data/trajectories_full`: translation median 1.62 m / p90 2.44 /
p99 4.14; aim change median 16.5° / p99 38.1 / max 39.2. Hence **±3 m / ±40°** — the inherited
±60° was 1.5× anything in the data.

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

Generated by `scripts/shared_manifest_table.py` straight from `runs/shared/*/summary.json`
(hand-copying ten numbers a row is how transcription errors get published).

| policy | mean imp | median | frac+ | d_end | d_best | @0.05 | @0.10 | @0.20 | shoot |
|---|---|---|---|---|---|---|---|---|---|
| **pi0.5** @300k | **0.612** | **0.535** | **0.92** | **0.254** | **0.226** | 11% | **30%** | **56%** | 26/64 |
| pi0.5 @270k | 0.599 | 0.487 | 0.92 | 0.266 | 0.223 | 11% | 31% | 55% | 24/64 |
| **GR00T** @210k | 0.400 | 0.408 | 0.78 | 0.465 | 0.349 | **12%** | 20% | 33% | 33/64 |
| **DP** (converged) | 0.357 | 0.294 | 0.72 | 0.508 | 0.425 | 0% | 5% | 17% | 42/64 |
| GR00T @130k | 0.235 | 0.238 | 0.64 | 0.631 | 0.479 | 5% | 16% | 28% | 33/64 |
| **UNIC** composition | −0.020 | −0.006 | 0.45 | 0.885 | 0.852 | 0% | 0% | 2% | 0/64 |
| **GPT-5** (goal image) | −0.038 | −0.006 | 0.48 | 0.904 | 0.774 | 0% | 2% | 5% | 0/64 |
| **GPT-5** (text goal) | −0.042 | +0.003 | 0.50 | 0.908 | 0.786 | 0% | 2% | 2% | 0/64 |
| **GPT-5** + measured profile | −0.133 | −0.033 | 0.44 | 0.999 | 0.712 | 0% | 6% | 16% | 0/64 |
| **PhotoFlow-style** (24 renders) | −0.984 | −1.157 | 0.20 | 1.850 | 0.758* | 0% | 0% | 2% | 0/64 |

\* PhotoFlow's `d_best` is `min(d_start, d_end)`, not the best viewpoint it visited — the global
branch records only the final pose in the trace. The best preview it actually rendered averages
**1.015**; see the regret analysis below.

Columns: `imp = d_start - d_end` (a no-op scores exactly 0); `frac+` = fraction of episodes that
improved at all; `d_best` = closest point on the path, so `d_best << d_end` means the policy passed
the shot and drifted off; `@tau` = fraction finishing within tau radians. Summaries in
`runs/shared/*/summary.json`, each carrying the manifest `key` so they join against a WAM run.

**Closure and delivery rank differently, so both belong in the table.** `improvement` says the
camera got closer; it does not say the requested photograph was taken. V12's
`calibrate_shot_threshold.py` finds ~66% of real frame pairs under d<0.10 carry the same six
cinematography words, ~0% past 0.60. The clearest case is DP vs GR00T **@130k**: DP wins every
closure column yet GR00T delivers the shot three times more often (16% vs 5% @0.10) — DP reliably
gets *moderately* closer and almost never lands the shot, GR00T either nails it or misses badly.
By **@210k** GR00T has overtaken DP on both (0.400 vs 0.357 improvement, 20% vs 5% @0.10), so that
particular split is a property of the undertrained checkpoint, not of the architecture; the general
point — that the two axes can disagree — stands on the @130k row.

**UNIC is the floor, and it is a no-op floor.** A published composition model driven in 3D scores
−0.020: 29/64 episodes improve, 35/64 get worse, net indistinguishable from leaving the camera
where it was, with 1/64 inside 0.20. That is structural rather than a tuning failure — UNIC has no
notion of the requested shot, so it optimises generic crop quality while the metric asks for a
*specified* viewpoint. Note it never declares `shoot` (0/64, it has no shoot head), so unlike DP
(42/64) its failure is "barely moved", not "moved wrongly then fired".

**pi0.5 wins both**, and its two checkpoints agree closely (0.612 / 0.599), so this is a property
of the policy rather than a lucky draw. It also stops well: `d_best` 0.226 vs `d_end` 0.254 means
almost no overshoot, and its success@0.10 is identical at best and end — while GR00T loses 19%→16%
by not stopping at its closest point. For reference, V12's own v5 @40k sits at `d_end` 0.35–0.40
with 17–25% inside 0.10.

### Every training-free row lands at or below the no-op floor

Four methods that never see a gradient — a published composition model, and a frontier VLM in
three conditioning modes — all score between −0.02 and −0.98, while every trained policy clears
+0.23. Goal-image conditioning (−0.038) is indistinguishable from the text brief (−0.042), so the
gap is not a goal-representation problem.

**The decisive difference is termination, not steering.** The GPT-5 agent declares `done` on
**0 of 64 episodes in all three modes** — the field is in the schema and it simply never sets it —
so it always ends on its last move rather than its best one:

| policy | d_best | d_end | drift | episodes drifting >0.1 | mean imp if it stopped at its closest point |
|---|---|---|---|---|---|
| GPT-5 + measured | 0.712 | 0.999 | **+0.287** | 52% | −0.133 → **+0.154** |
| GPT-5 (goal image) | 0.774 | 0.904 | +0.130 | 41% | −0.038 → +0.091 |
| GPT-5 (text goal) | 0.786 | 0.908 | +0.122 | 38% | −0.042 → +0.080 |
| GR00T @210k | 0.349 | 0.465 | +0.116 | 22% | +0.400 → +0.516 |
| DP | 0.425 | 0.508 | +0.083 | 17% | +0.357 → +0.441 |
| UNIC | 0.852 | 0.885 | +0.033 | 6% | −0.020 → +0.013 |
| pi0.5 @300k | 0.226 | 0.254 | +0.028 | 9% | +0.612 → +0.639 |

The measured-profile signal **works**: it buys the best `d_best` of any training-free row and 16%
of episodes inside 0.20 at some point, 8× the text row's 2%. Then it overshoots hardest in the
table and finishes with the worst `d_end` of anything. The information helps it approach and gives
it no way to stop. Every trained policy has a calibrated shoot head (pi0.5 26/64 and the lowest
drift in the table); the agent has none. The "stopped at its closest point" column is a diagnosis,
not a score — it requires knowing the metric, which no policy here gets.

UNIC's low drift is not a virtue: it barely moves at all (~0.4 m dollies, ~1° pans), so it has
nothing to drift from.

### PhotoFlow: the search is weak, the selection is worse

PhotoFlow is last by an order of magnitude, so all **1,536 previews** were scored geometrically
post-hoc (pure pose math on the logged `position`/`look_at`, no re-rendering):

| | mean d |
|---|---|
| start pose | 0.866 |
| best preview the search **visited** | 1.015 |
| preview the Reviewer **chose** | 1.856 |

**Regret (chosen − best visited) = +0.841.** The search reaches a viewpoint better than its start
in **33/64** episodes; `J` keeps it in **12/64**. Within 0.40 rad it visits a qualifying viewpoint
16 times and selects it 7 times. So the failure is two separate things, and the second dominates:

1. **Coverage** — best-visited (1.015) is worse on average than not moving (0.866). Proposing
   absolute camera states from a scene summary is much harder than nudging from the current view.
2. **Selection** — it discards good viewpoints it had already rendered, about half the time. Only
   0.20 of `J`'s weight is the geometric rule (`m1`+`m2`), and the `m6` instruction-alignment term
   that carries the goal ran ~0.0–0.18 across previews, so `J` was effectively driven by `m4`
   technical image quality. It reliably picks the *prettiest* frame, not the requested one.

This is a benchmark mismatch, not a broken reimplementation: the Director's blueprints are specific
and sensible, the Reflector kills regions and relocates, and the published constants are all in
place. PhotoFlow was built for free-form aesthetic missions where a VLM can judge "does this look
like the brief"; our goal is a precise geometric viewpoint a VLM cannot estimate from one frame.
Their own paper reports PhotoFlow beating Single-Step LLM .578 to .514 on `M_qs` — on *their*
metric. On ours the ordering inverts, and a 24× render budget does not rescue it.

Cost: 1,486 renders and **21,588 gpt-5 calls** for 64 episodes (~46× the three GPT-5 rows
combined), ~7 h across three GPUs. Sharded with `scripts/merge_shard_summaries.py`, which
recomputes aggregates from the pooled episodes and refuses partial or overlapping merges.

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

## Status (2026-10-07)

- **pi0.5** (`runs/pi05_fair/`) — **done.** 300k steps, 4.34 epochs, final loss 0.015, no
  divergence. Val-best is the final checkpoint (300k, 18.9 cm). This is the strongest baseline.
- **DP** (`runs/20260818_064526_diffusion_policy_dinov2/`) — **done**, early-stopped on val.
- **UNIC** (`runs/shared/unic/`) — **done**, 64/64, training-free.
- **GPT-5 agent** (`runs/shared/gpt5_{text,image,text_fb}/`) — **done**, 64/64 each, 156 calls each.
- **PhotoFlow-style** (`runs/shared/photoflow/`, merged from `photoflow_s{0,1,2}`) — **done**,
  64/64, 1,486 renders, 21,588 gpt-5 calls, 38 failures.
- **GR00T** (`runs/groot_fair/`) — **incomplete.** Two failures, in order:
  1. the first fair run diverged at the stock LR (grad-norm creep 0.8 → 6.3 → NaN at 168k; every
     checkpoint past ~130k is NaN, and 150k had already degraded to 320 cm);
  2. the retrain at half LR (5e-5) was stable but **died at step 210k of 300k** on 2026-09-04 when
     rank 1 took a SIGABRT and torchrun tore down rank 0. Cause not yet diagnosed; the 210k
     checkpoint is intact and predicts finite actions.

  The **210k evaluation has since landed** (`runs/shared/groot_210000/`, 2026-09-07; same manifest,
  64/64 unique episodes, `d_start` checksum 0.8656) and it moved the numbers a lot: mean improvement
  0.235 → **0.400**, @0.10 16% → 20%, `d_end` 0.631 → 0.465. GR00T now sits **above DP** on every
  column. So the long-standing "@130k" row was understating GR00T substantially; both rows are kept
  above to show the checkpoint progression. It remains budget-unequal against pi0.5's 300k — but by
  210k/300k rather than 130k/300k.
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
