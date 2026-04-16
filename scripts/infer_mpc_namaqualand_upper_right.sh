#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Namaqualand scene — "오른쪽 위에서 내려다보는 느낌, 중앙, 50% 크기"
RUN_DIR="outputs/Namaqualand_namaqualand_v3_260331_054741"
MODEL_PATH="runs/20260403_151944_qwen35_vl_2b_1xh200_with_c2o_5k/final"
BLENDER_BIN="blender/blender"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-96}"
INITIAL_SEED="${INITIAL_SEED:-721}"

# Weights: bbox framing + camera-object orientation
DEFAULT_SCORE_WEIGHTS='{"bbox_occupancy_ratio":2.0,"bbox_margin_top":1.0,"bbox_margin_bottom":1.0,"bbox_margin_left":1.0,"bbox_margin_right":1.0,"bbox_aspect_ratio":1.0,"bbox_centroid_offset":2.0,"camera_to_object_fx":1.0,"camera_to_object_fy":1.0,"camera_to_object_fz":1.0,"camera_to_object_ux":1.0,"camera_to_object_uy":1.0,"camera_to_object_uz":1.0}'
SCORE_WEIGHTS_JSON="${SCORE_WEIGHTS_JSON:-$DEFAULT_SCORE_WEIGHTS}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

# Target:
#   bbox: 중앙 배치, 50% occupancy
#   camera_to_object: 오른쪽 위에서 내려다보는 구도
#     fy>0=정면: (0.5, +0.7, -0.3) 정면+오른쪽+약간 위에서
#     object up: (0.0, -0.4, 0.9)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python scripts/infer_mpc_blender.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --config configs/qwen35_vl_2b_1xh200_with_c2o_5k.yaml \
  --blender_bin "${BLENDER_BIN}" \
  --initial_seed "${INITIAL_SEED}" \
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
  --target_json '{"bbox_centroid_offset":0.0,"bbox_occupancy_ratio":0.5,"camera_to_object_fx":0.5,"camera_to_object_fy":0.7,"camera_to_object_fz":-0.3,"camera_to_object_ux":0.4,"camera_to_object_uy":0.1,"camera_to_object_uz":0.9}'
