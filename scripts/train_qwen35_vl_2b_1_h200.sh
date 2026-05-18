#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python scripts/train.py \
  --config configs/qwen35_vl_2b_1xh200.yaml
