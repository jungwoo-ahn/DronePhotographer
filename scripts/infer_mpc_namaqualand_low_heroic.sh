#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 아래에서 올려다보는 — 히어로 샷 (low angle)
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

# 아래에서 올려다봄 (정면): fy>0=정면, 데이터 확인 fy≈+0.8 fz≈0 uy≈0 uz≈+1.0
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
  --target_json '{"bbox_centroid_offset":0.0,"bbox_occupancy_ratio":0.45,"camera_to_object_fx":0.0,"camera_to_object_fy":0.8,"camera_to_object_fz":0.0,"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}'
