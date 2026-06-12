# Cosmos policy training — handoff (jungwoo)

State of `005-cosmos-policy-integration` as of 2026-06-12. Everything below was
smoke-tested on 2× RTX A6000 (48 GB): frozen-backbone DDP smoke (200 iters,
completed) and a full-finetune DDP run (1000 iters). This doc is what you need
to run training on your server.

## 0. TL;DR

```bash
# put the v7 render output under data/trajectories (placement dirs, scored)
ln -s /path/to/outputs/v7_renders/* data/trajectories/

# single GPU
PYTHONPATH=. python scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml

# multi-GPU (DDP) — one process per GPU
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml

# tensorboard
tensorboard --logdir runs --port 6006 --bind_all
```

Outputs land in `runs/<timestamp>_<run_name>/` — `config.yaml` (resolved),
`train.log`, `tb/` (TensorBoard), `ckpt_last.pt` + `ckpt_best.pt` (best = lowest
loss-EMA at a save point). Checkpoints carry the raw policy state dict (no DDP
`module.` prefix) and the `action_scale` buffer.

## 1. Environment

- `pip install -r requirements.txt` — the load-bearing pins: `torch>=2.7`,
  `diffusers>=0.35.2`, `transformers>=4.57`, `accelerate`, `tensorboard`.
- **HF auth**: `nvidia/Cosmos-Predict2.5-2B` is gated. `hf auth login` with an
  account that accepted the NVIDIA license. First run downloads ~5 GB
  (transformer + Wan VAE only; the Qwen text encoder is never downloaded).
- **Text anchor**: `assets/text_anchor.pt` (~100 MB) is gitignored. If missing,
  rebuild once (downloads Cosmos-Reason1 / Qwen2.5-VL, needs a big GPU):
  ```bash
  PYTHONPATH=. python scripts/build_text_anchor.py \
      --prompt "A smooth aerial drone shot. The camera moves continuously in one take, deliberately adjusting its viewpoint." \
      --output assets/text_anchor.pt
  ```
  Or just copy the file from this server — it's deterministic for a given prompt.
- Tests (no GPU/diffusers needed): `python -m pytest tests/policy/` — 49 tests.

## 2. Data

`data.annotation_roots` (default `data/trajectories`) must hold v7 placement
dirs: `<scene>__<object>/` with `data.json`, `renders/pair_PP_frame_FF.jpg`,
`done.flag`, `scored.flag`. Only scored placements yield samples. The loader
globs `**/data.json` recursively, so symlinking the whole render output dir in
is fine.

Scale note: 1945 scored placements at `stride: 1` ≈ 500k windows. Dataset build
parses every `data.json` at startup — expect a few minutes before iter 1.

## 3. What one training sample is

A `chunk_size=8`-step window over one `accepted_pairs[i].trajectory_32f`:

- `state_image` (start frame) + `next_state_image` (end frame) — ALOHA-style
  T_img=2; world head predicts the end frame.
- `action_chunk` (8 × 5D) — the consecutive camera deltas across the window.
- `goal_vec` — 8-key V5 profile of the **goal frame**, which is drawn uniformly
  from `[end_frame, 31]` per `__getitem__` (HER-"future" relabeling,
  `data.goal_sampling: uniform_future`; `"end"` restores the legacy fixed
  offset). Action chunk and next-frame target stay anchored to the window;
  only goal_vec + value follow the drawn frame.
- `value_target` — `−pose_distance(start, goal)` in radians. **Always ≤ 0**
  (0 = start already achieves the goal framing). The positive `value=` in the
  train log is the loss component, not the target.

Goal candidates that hit the scorer's off-screen sentinel
(`occupancy==0 && bbox_y_offset==0`) are excluded; windows with no valid
candidate are dropped.

## 4. Validation (new)

The flow-matching train loss draws a random sigma per batch — per-iter loss is
noisy by construction and is NOT a quality signal. Validation fixes that:

- **Fixed-sigma val loss**: every val batch evaluated at sigma ∈
  {0.1,…,0.9} with a fixed noise seed → `val/flow_loss_sigma_*`,
  `val/flow_loss_mean`. Comparable across checkpoints.
- **Sampler metrics**: real Euler sampling (`val_sample_steps: 8`) on val
  windows → `val/action_mse` (vs ground-truth chunk, normalized space) and
  `val/value_mae`. These are the sample-quality numbers.
- Runs on rank 0 every `trainer.val_iter` (200) iters; logged to train.log
  (`iter=N VAL …`) and TB.

**Split** — frozen scene manifest, `configs/policy/val_scenes.txt`:
2× Forest-field (same asset family — held out together to avoid lookalike
leakage), Modern-Residential-Building-Facade, Desert-Roadside-Repair-Garage.
That's 73/1945 placements (3.8%) — chosen as the smallest scenes so the
unseen-scene split costs almost no training data. Properties:

- The val set is **pinned forever**: new placements/scenes always go to train.
  Manifest membership is by scene name; re-rendering or adding data never
  moves anything across the split.
- Don't edit the manifest after comparable experiments have started — that
  redefines the split.
- Alternative hash split (`val_pair_stride` + `val_split_level:
  pair|placement|scene|object`) exists for dev; `val_names` overrides it.
- `val_max_samples: 64` caps val cost, subsampled evenly across scenes.

## 5. Multi-GPU

`torchrun` is auto-detected (WORLD_SIZE env). Per-GPU `batch_size` comes from
the config → effective batch = `batch_size × grad_accum × nproc`. Details that
already work, so don't re-solve them: rank-0-staggered HF download, single
rank-0 run dir (timestamp broadcast), `DistributedSampler` + per-epoch
reshuffle, `no_sync()` on grad-accum micro-steps, per-rank seeds for
decorrelated sigma draws, rank-0-only logging/val/checkpointing.

Observed on 2× A6000 48 GB, 480×720, batch 1: frozen backbone 0.14 it/s/rank
(DDP overhead ≈ 0); full finetune 0.05 it/s/rank, fits in 48 GB **only with**
`gradient_checkpointing: true` (default when unfrozen).

## 6. Knobs and gotchas

- `--max_iter / --max_samples / --warmup_iter` CLI overrides exist.
  **`warmup_iter: 1000` in the config assumes a long run** — for short runs
  pass `--warmup_iter 100` or the whole run sits inside warmup at low lr.
- The loss is a single joint flow-matching velocity MSE, split by latent
  position into `world/action/value` parts (weights `loss.lambda_*`).
- Inference/eval: `scripts/eval_cosmos_policy.py --checkpoint … --chunk_size 8`
  (must match training). `--render` is still stubbed.
- `ACTION_SCALE` is provisional (p99 over the 392-trajectory Stage-1 sample) —
  worth recomputing over the full dataset; the trained value is persisted in
  the checkpoint either way.
- VAE latent cache (`scripts/encode_vae_latents.py`) is still not consumed by
  the trainer; images are encoded on the fly.
- `src/policy/README.md` predates the Qwen anchor + flow-matching + HER changes
  in places; where it disagrees with this doc or the code, the code wins.
