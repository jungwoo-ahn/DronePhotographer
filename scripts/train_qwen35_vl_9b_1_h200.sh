#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python scripts/train_qwen25_vl.py \
  --config configs/qwen35_vl_9b_1xh200.yaml
