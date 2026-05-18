#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 오버숄더 — 뒤쪽 옆에서, 약간 위에서, 어깨 너머로 찍는 느낌
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

# 오버숄더: fy=-0.5 (뒤쪽), fx=+0.6 (왼쪽에서), fz=-0.2 (약간 위에서)
# 피사체를 오른쪽에 배치, 왼쪽에 시선 방향 여백
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
  --target_json '{"center_x":0.6,"center_y":0.45,"occupancy":0.35,"aspect_ratio":1.3,"camera_to_object_fx":0.6,"camera_to_object_fy":-0.5,"camera_to_object_fz":-0.2,"camera_to_object_ux":0.0,"camera_to_object_uy":-0.2,"camera_to_object_uz":0.95}'
