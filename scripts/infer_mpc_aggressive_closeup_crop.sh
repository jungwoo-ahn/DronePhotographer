#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

RUN_DIR="outputs/DogWalk_v2_10k_260309_101152"
MODEL_PATH="runs/20260312_150649_qwen35_vl_2b_1xh200/checkpoints/checkpoint-14000"
BLENDER_BIN="blender/blender"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-96}"
INITIAL_SEED="${INITIAL_SEED:-721}"
DEFAULT_SCORE_WEIGHTS='{"bbox_occupancy_ratio":4.0,"bbox_margin_top":1.8,"bbox_margin_bottom":3.5,"bbox_margin_left":1.3,"bbox_margin_right":1.3,"bbox_aspect_ratio":1.8,"bbox_centroid_offset":0.8}'
SCORE_WEIGHTS_JSON="${SCORE_WEIGHTS_JSON:-$DEFAULT_SCORE_WEIGHTS}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

CUDA_VISIBLE_DEVICES=5 python scripts/infer_mpc_blender.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --config configs/qwen35_vl_2b_1xh200.yaml \
  --blender_bin "${BLENDER_BIN}" \
  --initial_seed "${INITIAL_SEED}" \
  --num_steps 16 \
  --translation_values_m=-0.12,-0.06,0,0.06,0.12 \
  --rotation_values_deg=-3,0,3 \
  --max_translation_norm_m 0.18 \
  --max_rotation_norm_deg 4.5 \
  --max_candidates 720 \
  --candidate_batch_size "${CANDIDATE_BATCH_SIZE}" \
  --max_new_tokens 128 \
  --score_weights_json "${SCORE_WEIGHTS_JSON}" \
  --translation_penalty_weight 0.0 \
  --rotation_penalty_weight 0.0 \
  --blender_threads "${BLENDER_THREADS}" \
  --target_json '{"bbox_occupancy_ratio":0.67,"bbox_margin_top":0.02,"bbox_margin_bottom":0.0,"bbox_margin_left":0.16,"bbox_margin_right":0.16,"bbox_aspect_ratio":0.69}'
