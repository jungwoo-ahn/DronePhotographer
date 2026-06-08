# `src/policy/` — Cosmos goal-conditioned policy

Operational runbook: where data goes, how to set up, train, and evaluate. For the
**architecture / tensor contract** see [`cosmos/COSMOS_API.md`](cosmos/COSMOS_API.md).
For the data-generation pipeline see `docs/v7_handoff_jooyeol.md` on branch
`v7_data_for_cosmos_policy`.

## What this is

A goal-conditioned video world-action policy on the frozen Cosmos-Predict2.5-2B
backbone. One training sample is a K-step window over a rendered camera
trajectory: it learns to jointly predict the next frames (world), the K-step
camera action chunk (policy), and a value, conditioned on a shot-profile goal
vector. Only the goal conditioner + the action/value latent heads train; the
backbone is frozen.

```
src/policy/
  common/        goal_space, action_repr (5D), annotations (v7 iterator), reward, dataset_base
  cosmos/        vae, conditioner (T5-anchor + zero-init prefix), model, trainer, edm, COSMOS_API.md
  vla/, diffusion_policy/   placeholders for future families
```

## 1. Data layout

The dataset reads the **v7 Stage-2/3** output: one directory per (scene, object)
placement, each with a `data.json` and a `renders/` folder.

```
outputs/v7_stage2_renders/
  <scene>__<object>/
    data.json                         # accepted_pairs[].trajectory_32f + render_records[][].scores
    renders/pair_<pp>_frame_<ff>.jpg  # K_accepted × 32 JPEGs
    done.flag                         # Stage 2 (render) complete
    scored.flag                       # Stage 3 (V5 scoring) complete
  <scene>__<object>/
    ...
```

Point `data.annotation_roots` in the config at the directory holding the
placement subdirs (it recursively globs `**/data.json`). Image paths inside
`data.json` are resolved relative to each placement directory automatically — no
separate image root needed.

**A placement is usable only once `scored.flag` exists** — the goal vector comes
from `render_records[i][j].scores`, written by Stage 3. Placements missing scores
yield NaN goals and are silently skipped by the loader.

## 2. One-time setup

```bash
# (a) deps — needs diffusers for the real backbone (not in the base env yet)
pip install -r requirements.txt
pip install --no-build-isolation flash-attn>=2.7.3

# (b) build the fixed T5 anchor (loads T5-11B once, ~22 GB, ~3 min on an H100).
#     Produces assets/t5_anchor.pt, which the conditioner loads every run.
python scripts/build_t5_anchor.py \
    --prompt "A drone cinematography" \
    --output assets/t5_anchor.pt
```

The T5-11B encoder is used **only** by step (b) and never again — the goal
conditioner is a small learnable projection on top of the frozen anchor (see
COSMOS_API.md § "EDM" and § conditioner).

## 3. Train

```bash
# Smoke run (50 iters, 32 samples) — verifies the data path end to end.
python scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml --debug

# Full run.
python scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml
```

Outputs land in `runs/<timestamp>_<run_name>/` (checkpoints + a copy of the
resolved config). Per-iter logs print the joint loss broken into `world` /
`action` / `value` components.

Key config knobs (`configs/policy/cosmos_2b.yaml`):

| Block | Knob | Meaning |
|---|---|---|
| `data` | `chunk_size` | actions predicted per diffusion sample (default 8) |
| `data` | `stride` | window stride along each 32-frame trajectory |
| `data` | `goal_score_keys` | which V5 keys form the goal vector (default all 8) |
| `loss` | `lambda_world/action/value` | per-component loss weights |
| `edm` | `use_balanced_two_heads`, `high/low_sigma_ratio` | σ-sampling (cosmos-policy `BALANCED_TWO_HEADS_V1`) |
| `conditioner` | `anchor_path` | path to `t5_anchor.pt` from step 2b |

## 4. (Optional) precompute VAE latents

```bash
python scripts/encode_vae_latents.py \
    --annotation_roots outputs/v7_stage2_renders \
    --output runs/vae_cache/v7.pt
```

⚠️ **Not yet wired into the trainer** — it still encodes on the fly. This caches
per-frame latents for when a cache-load path is added. See the script docstring.

## 5. Evaluate

```bash
python scripts/eval_cosmos_policy.py \
    --checkpoint runs/<ts>_cosmos_2b/ckpt_last.pt \
    --start_annotation outputs/v7_stage2_renders/<placement>/data.json \
    --target configs/policy/targets/centered_medium.yaml \
    --chunk_size 8        # must match training
```

Prints the predicted (denormalized) action chunk + value, applies the first
action to the start pose, and reports a pose-proxy goal distance.

**Two gotchas:**
- `--chunk_size` must equal the training `chunk_size`.
- The `target:` YAML must list the **same goal keys, same order** as training
  (`data.goal_score_keys`) — `goal_dim` is inferred from it, and a mismatch
  silently loads a wrong-shaped conditioner. Use the new V5 integer/pixel schema
  (`configs/policy/targets/centered_medium.yaml`). The legacy
  `configs/inference/*.yaml` (fractional units, `mpc:` block) are **not**
  compatible.
- `--render` (Blender rollout) is still stubbed.

## 6. Tests

```bash
python -m pytest tests/policy/          # 89 tests, no GPU / no diffusers needed
```

The mock-backbone tests (`test_cosmos_model_mock.py`, `test_v7_integration.py`)
exercise the full data → encode → EDM loss → sample path without the 2B model.

## Value target

`value_target = −geometric_profile_distance(start_profile, goal_profile)`
(`src/policy/common/reward.py`). The profile is mapped back to the ~5 geometric
DOF of the camera-subject configuration — viewing direction (az, el), apparent
size, and optical-axis aim — and the distance is a **weight-free product metric**
in radians: great-circle angle on the viewing sphere (handles azimuth cyclicity +
polar degeneracy) + Δ angular size + Δ aim. `atan` bounds off-frame/huge-bbox
values gracefully. It's scale-invariant, matching the profile's scene-agnostic
design, and non-trivial (start ≠ goal). No per-key weights to tune.

## Known gaps / provisional values

- **`ACTION_SCALE` is provisional** — p99 magnitudes measured over the 392-trajectory
  Stage-1 sample (`src/policy/common/action_repr.py`). Recompute over the full
  rendered dataset and swap the constant.
- **Goal ranges assume 1024×768 renders** (`RENDER_WIDTH/HEIGHT` in
  `goal_space.py`). Change if resolution changes.
- **Azimuth is cyclic** but normalized linearly (seam at 0°/360°). sin/cos
  encoding is the fix if needed.
- **VAE latent cache** not consumed by the trainer yet (§4).
- **Backbone call convention** (`DiffusersStyleAdapter`) assumes the diffusers
  wrapping of Cosmos-Predict2.5; if loading the raw cosmos transformer, switch to
  `adapter="cosmos_native"` (see `model.py`).
