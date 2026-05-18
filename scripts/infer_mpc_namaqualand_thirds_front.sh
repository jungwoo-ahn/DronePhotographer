#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 삼분할 구도 — 오른쪽 위 교차점에 피사체, 정면 눈높이
RUN_DIR="outputs/Namaqualand_namaqualand_v3_260331_054741"
MODEL_PATH="runs/20260403_151944_qwen35_vl_2b_1xh200_with_c2o_5k/final"
BLENDER_BIN="blender/blender"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-96}"

DEFAULT_SCORE_WEIGHTS='{"bbox_occupancy_ratio":2.0,"bbox_margin_top":1.0,"bbox_margin_bottom":1.0,"bbox_margin_left":1.0,"bbox_margin_right":1.0,"bbox_aspect_ratio":1.0,"bbox_centroid_offset":2.0,"camera_to_object_fx":1.0,"camera_to_object_fy":1.0,"camera_to_object_fz":1.0,"camera_to_object_ux":1.0,"camera_to_object_uy":1.0,"camera_to_object_uz":1.0}'
SCORE_WEIGHTS_JSON="${SCORE_WEIGHTS_JSON:-$DEFAULT_SCORE_WEIGHTS}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

# 삼분할 오른쪽 위: center=(0.667,0.333), 25% occupancy, 정면 eye-level
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python scripts/infer_mpc_blender.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --config configs/qwen35_vl_2b_1xh200_with_c2o_5k.yaml \
  --blender_bin "${BLENDER_BIN}" \
  --initial_seed 721 \
  --num_steps 50 \
  --translation_values_m=-0.2,-0.1,0,0.1,0.2 \
  --rotation_values_deg=-5,0,5 \
  --max_translation_norm_m 0.3 \
  --max_rotation_norm_deg 7.5 \
  --max_candidates 720 \
  --candidate_batch_size "${CANDIDATE_BATCH_SIZE}" \
  --max_new_tokens 256 \
  --score_weights_json "${SCORE_WEIGHTS_JSON}" \
  --translation_penalty_weight 0.0 \
  --rotation_penalty_weight 0.0 \
  --blender_threads "${BLENDER_THREADS}" \
  --disable_roll \
  --target_json '{"center_x":0.667,"center_y":0.333,"occupancy":0.25,"aspect_ratio":1.0,"camera_to_object_fx":0.0,"camera_to_object_fy":1.0,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}'
