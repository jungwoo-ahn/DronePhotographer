#!/usr/bin/env bash
# Fine-tune pretrained GR00T N1.7 (LeRobot `groot`) on our LeRobot export.
#
# 2nd real-pretrained-VLA baseline (after pi0.5). N1.7 (NOT N1.5 — LeRobot 0.6 dropped
# N1.5). VLM backbone = Cosmos-Reason2-2B; diffusion (flow) action head.
#
# Prereqs:
#   1. conda env `vla_groot` (clone of `vla` + peft/diffusers/timm/dm-tree/decord)
#   2. HF: accept the gated backbone nvidia/Cosmos-Reason2-2B (gated=auto -> instant) with
#      the logged-in account. GR00T loads it on first use, else 403.
#   3. dataset built by scripts/export_lerobot.py with **--state-dim 0** (NO observation.state:
#      a degenerate all-zeros proprio channel risks div-by-zero in GR00T normalization).
#
# Memory: vision + LLM are FROZEN by default (tune_visual=false, tune_llm=false); only the
# diffusion head + projector train. Fits one 49GB GPU. NEVER use GPU 0 / a shared GPU.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/nas5/jooyeolyun/datasets/drone_data/lerobot_groot_v1}"
REPO_ID="${REPO_ID:-drone/lerobot_groot_v1}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"
OUT="${OUT:-runs/groot_drone}"

NPROC="${NPROC:-1}"
ARGS=(
  --dataset.repo_id="${REPO_ID}"
  --dataset.root="${DATA_ROOT}"
  --policy.type=groot
  --policy.base_model_path=nvidia/GR00T-N1.7-3B
  --policy.embodiment_tag=new_embodiment
  --policy.device=cuda
  --policy.use_bf16=true
  --policy.chunk_size=8
  --policy.n_action_steps=8
  --policy.push_to_hub=false
  --batch_size="${BATCH}"
  --num_workers=8
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ:-5000}"
  --log_freq=100
  --env_eval_freq=0
  --eval_steps=0
  --wandb.enable=false
  --seed=1000
  --output_dir="${OUT}"
  --job_name=groot_drone
)

if [ "${NPROC}" -gt 1 ]; then
  # DDP: GR00T (~38GB/GPU at batch 8, fp32 params + bf16 autocast) FITS one 49GB GPU, so
  # REPLICATE per rank + split the batch. Leaving every sharding field default makes plain
  # torchrun auto-fill dp_replicate = world_size (out-of-the-box DDP). --batch_size is
  # PER-RANK, so effective batch = BATCH * NPROC. GR00T's cosine scheduler already spans the
  # full --steps horizon (no LR-floor fix needed, unlike pi05).
  # distinct --master-port per concurrent torchrun job on the same host (default 29500 clashes
  # with another running DDP job -> EADDRINUSE). Override with MASTER_PORT for concurrent runs.
  exec torchrun --nproc-per-node="${NPROC}" --master-port="${MASTER_PORT:-29500}" \
    "$(which lerobot-train)" "${ARGS[@]}" "$@"
else
  exec lerobot-train "${ARGS[@]}" "$@"
fi
