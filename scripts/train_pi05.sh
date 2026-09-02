#!/usr/bin/env bash
# Fine-tune pretrained pi0.5 (LeRobot `pi05`, PyTorch — no JAX) on our LeRobot export.
#
# Baseline: a REAL pretrained VLA (pi0.5, ~3B, PaliGemma + Gemma action expert) fine-tuned
# on our (start-frame image, natural-language shot-profile goal, 10D camera action) data.
# Replaces the hand-rolled Qwen3-VL "pi0-style" reimplementation.
#
# Prereqs:
#   1. conda env `vla` (lerobot 0.4.4, torch 2.10+cu128, ffmpeg 7 for torchcodec decode)
#   2. HF: the logged-in account must have ACCEPTED the gated google/paligemma-3b-pt-224
#      license (pi05_base pulls its tokenizer/backbone). Without it, load 403s.
#   3. dataset built by scripts/export_lerobot.py (LeRobot v3.0, meta/stats.json w/ q01/q99).
#
# Memory: full fine-tune of pi0.5 needs >70GB (2 GPUs). This recipe uses
# train_expert_only + freeze_vision_encoder + grad-checkpoint + bf16 to fit ONE 49GB GPU.
# Override GPU via CUDA_VISIBLE_DEVICES; NEVER use GPU 0 or share a foreign process.
set -euo pipefail

# The gated google/paligemma-3b-pt-224 tokenizer is vendored into the HF cache (verified
# byte-identical: vocab 257152, sha256 8986bb4f…). Until the account is granted official
# access, run offline so from_pretrained("google/paligemma-3b-pt-224") uses the cache
# instead of 403-ing. pi05_base weights + our dataset are already local/cached.
# Once official access lands, these can be dropped (or left; harmless).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATA_ROOT="${DATA_ROOT:-/home/nas5/jooyeolyun/datasets/drone_data/lerobot_pi05_v1}"
REPO_ID="${REPO_ID:-drone/lerobot_pi05_v1}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-32}"
OUT="${OUT:-runs/pi05_drone}"

# STATE forced to IDENTITY: our observation.state is zeros (no proprio — image + language
# goal only, matching the DP / custom baselines), so quantile/mean-std would divide by ~0.
# ACTION stays QUANTILES (pi05 default; our stats.json carries q01/q99). VISUAL IDENTITY.
# ACTION_NORM: QUANTILES (pi05 default) or MEAN_STD (LIBERO recipe — more robust for the
# near-identity framewise rot6d, whose signal is tiny deviations quantiles can distort).
ACTION_NORM="${ACTION_NORM:-QUANTILES}"
NORM="{\"VISUAL\":\"IDENTITY\",\"STATE\":\"IDENTITY\",\"ACTION\":\"${ACTION_NORM}\"}"

# FREEZE=true (default): expert-only fine-tune, fits ONE 49GB GPU but UNDERFITS our novel
# camera-action space (too-conservative actions). FREEZE=false: full fine-tune (unfreeze the
# VLM) — pi0.5's primary recipe; needs >70GB so run multi-GPU with NPROC=2 (FSDP dp_shard).
FREEZE="${FREEZE:-true}"
NPROC="${NPROC:-1}"

ARGS=(
  --dataset.repo_id="${REPO_ID}"
  --dataset.root="${DATA_ROOT}"
  --policy.type=pi05
  --policy.pretrained_path=lerobot/pi05_base
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing="${GC:-true}"
  --policy.train_expert_only="${FREEZE}"
  --policy.freeze_vision_encoder="${FREEZE}"
  --policy.normalization_mapping="${NORM}"
  --policy.chunk_size="${CHUNK:-8}"
  --policy.n_action_steps=8
  --policy.push_to_hub=false
  --batch_size="${BATCH}"
  --num_workers=8
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ:-5000}"
  --log_freq=100
  --env_eval_freq=0
  --wandb.enable=false
  --seed=1000
  --output_dir="${OUT}"
  --job_name=pi05_drone
)

if [ "${NPROC}" -gt 1 ]; then
  if [ "${PARALLEL:-ddp}" = "ddp" ]; then
    # DDP (default): the 3B model FITS one 49GB GPU at per-GPU batch 8 (~36GB w/ GC+bf16),
    # so REPLICATE it per rank and split the batch — no FSDP, no DTensor mixing bug.
    # Per lerobot/configs/parallelism.py: leaving every sharding field at default makes
    # plain torchrun auto-fill dp_replicate = world_size (out-of-the-box DDP). --batch_size
    # is PER-RANK, so effective batch = BATCH * NPROC.
    # distinct --master-port per concurrent torchrun job (default 29500 clashes -> EADDRINUSE).
    exec torchrun --nproc-per-node="${NPROC}" --master-port="${MASTER_PORT:-29500}" \
      "$(which lerobot-train)" "${ARGS[@]}" "$@"
  else
    # FSDP shard the 3B model across NPROC GPUs (only needed if it does NOT fit one GPU).
    # PI05Policy declares no _fsdp_wrap_modules. Size-based auto-wrap fails (it wraps a
    # forward-less ModuleList), so name the actual transformer-layer classes explicitly:
    # _PiGemmaDecoderLayerBase (Gemma LM + action expert, x36) + SiglipEncoderLayer (vision, x27).
    ARGS+=( --parallelism.dp_shard="${NPROC}" --accelerator.mixed_precision=bf16
            '--accelerator.fsdp.wrap_modules=["_PiGemmaDecoderLayerBase","SiglipEncoderLayer"]' )
    exec torchrun --nproc-per-node="${NPROC}" "$(which lerobot-train)" "${ARGS[@]}" "$@"
  fi
else
  exec lerobot-train "${ARGS[@]}" "$@"
fi
