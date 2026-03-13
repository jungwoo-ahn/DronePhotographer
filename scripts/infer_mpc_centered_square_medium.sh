#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RUN_DIR="outputs/DogWalk_v2_10k_260309_101152"
MODEL_PATH="runs/20260312_150649_qwen35_vl_2b_1xh200/checkpoints/checkpoint-13500"
BLENDER_BIN="blender/blender"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-96}"

CUDA_VISIBLE_DEVICES=1 python scripts/infer_mpc_blender.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --config configs/qwen35_vl_2b_1xh200.yaml \
  --blender_bin "${BLENDER_BIN}" \
  --num_steps 16 \
  --translation_values_m=-0.12,-0.06,0,0.06,0.12 \
  --rotation_values_deg=-3,0,3 \
  --max_translation_norm_m 0.18 \
  --max_rotation_norm_deg 4.5 \
  --max_candidates 720 \
  --candidate_batch_size "${CANDIDATE_BATCH_SIZE}" \
  --max_new_tokens 128 \
  --translation_penalty_weight 0.0 \
  --rotation_penalty_weight 0.0 \
  --target_json '{"center_x":0.5,"center_y":0.5,"occupancy":0.3,"aspect_ratio":1.0}'
