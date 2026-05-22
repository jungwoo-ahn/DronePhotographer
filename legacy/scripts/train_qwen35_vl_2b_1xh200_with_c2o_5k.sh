#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/train.py \
  --config configs/qwen35_vl_2b_1xh200_with_c2o_5k.yaml
