#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Usage -----------------------------------------------------------
usage() {
    cat <<'USAGE'
End-to-end MPC pipeline: natural language → target JSON → Blender-in-the-loop inference

Usage:
  bash scripts/run_e2e_mpc.sh "front eye-level, centered, medium shot"

Environment overrides:
  CUDA_VISIBLE_DEVICES   GPU indices          (default: 0,1)
  RUN_DIR                scene run directory
  MODEL_PATH             trained model checkpoint
  CONFIG                 training config yaml
  BLENDER_THREADS        CPU threads for Blender  (default: 4)
  CANDIDATE_BATCH_SIZE   VLM batch size           (default: 192)
  TARGET_JSON            skip Stage 1, use this target directly
USAGE
    exit 1
}

DESCRIPTION="${1:-}"
if [[ -z "${DESCRIPTION}" && -z "${TARGET_JSON:-}" ]]; then
    usage
fi

# --- Configurable defaults -------------------------------------------
RUN_DIR="${RUN_DIR:-outputs/Namaqualand_namaqualand_v3_260401_024633}"
MODEL_PATH="${MODEL_PATH:-runs/20260418_090117_qwen35_vl_2b_4xh200_with_c2o_5k_v2/final_converted}"
CONFIG="${CONFIG:-configs/qwen35_vl_2b_4xh200_with_c2o_5k_v2.yaml}"
BLENDER_BIN="${BLENDER_BIN:-blender/blender}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-192}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

DEFAULT_SCORE_WEIGHTS='{"bbox_occupancy_ratio":1.0,"bbox_margin_top":1.0,"bbox_margin_bottom":1.0,"bbox_margin_left":1.0,"bbox_margin_right":1.0,"bbox_aspect_ratio":1.0,"bbox_centroid_offset":1.0,"body_in_frame_ratio":1.0,"camera_to_object_fx":1.0,"camera_to_object_fy":1.0,"camera_to_object_fz":1.0,"camera_to_object_ux":1.0,"camera_to_object_uy":1.0,"camera_to_object_uz":1.0}'
SCORE_WEIGHTS_JSON="${SCORE_WEIGHTS_JSON:-$DEFAULT_SCORE_WEIGHTS}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

# --- Preflight checks ------------------------------------------------
errors=0
if [[ ! -x "${BLENDER_BIN}" ]]; then
    echo "ERROR: Blender not found at ${BLENDER_BIN}" >&2
    echo "  Run: bash scripts/install_blender.sh" >&2
    errors=1
fi
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: Scene run directory not found: ${RUN_DIR}" >&2
    errors=1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Model checkpoint not found: ${MODEL_PATH}" >&2
    errors=1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}" >&2
    errors=1
fi
if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required but not found" >&2
    errors=1
fi
if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI is required but not found (needed for --backend claude-cli)" >&2
    errors=1
fi
if (( errors )); then
    exit 1
fi

# --- Stage 1: Generate target JSON from natural language --------------
if [[ -n "${TARGET_JSON:-}" ]]; then
    echo "=== Stage 1: Skipped (using provided TARGET_JSON) ==="
    echo "Target: ${TARGET_JSON}"
    echo ""
else
    echo "=== Stage 1: Generating target JSON ==="
    echo "Description: ${DESCRIPTION}"
    echo ""

    GENERATE_OUTPUT="$(python scripts/generate_target.py \
        --backend claude-cli \
        --run_dir "${RUN_DIR}" \
        --model_path "${MODEL_PATH}" \
        --config "${CONFIG}" \
        "${DESCRIPTION}")"

    TARGET_JSON="$(echo "${GENERATE_OUTPUT}" | jq -c '.target_json')"

    if [[ -z "${TARGET_JSON}" || "${TARGET_JSON}" == "null" ]]; then
        echo "ERROR: Failed to extract target_json from generate_target.py output" >&2
        echo "Raw output:" >&2
        echo "${GENERATE_OUTPUT}" >&2
        exit 1
    fi

    echo "Generated target:"
    echo "${GENERATE_OUTPUT}" | jq '.target_json'
    echo ""
    echo "Human-readable:"
    echo "${GENERATE_OUTPUT}" | jq -r '.human_readable'
    echo ""
fi

# --- Stage 2: Run Blender-in-the-loop MPC inference -------------------
echo "=== Stage 2: Running MPC inference (GPUs: ${CUDA_VISIBLE_DEVICES}) ==="
echo ""

python scripts/infer_mpc_blender.py \
  --use_vllm \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --config "${CONFIG}" \
  --blender_bin "${BLENDER_BIN}" \
  --initial_seed 721 \
  --num_steps 50 \
  --translation_values_m=-0.5,-0.25,0,0.25,0.5 \
  --rotation_values_deg=-8,0,8 \
  --max_translation_norm_m 0.5 \
  --max_rotation_norm_deg 12 \
  --max_candidates 720 \
  --candidate_batch_size "${CANDIDATE_BATCH_SIZE}" \
  --max_new_tokens 256 \
  --score_weights_json "${SCORE_WEIGHTS_JSON}" \
  --translation_penalty_weight 0.0 \
  --rotation_penalty_weight 0.0 \
  --blender_threads "${BLENDER_THREADS}" \
  --disable_roll \
  --disable_adaptive_step \
  --lookahead_top_k 0 \
  --target_json "${TARGET_JSON}"

echo ""
echo "=== Pipeline complete ==="
echo "Description: ${DESCRIPTION:-"(provided TARGET_JSON)"}"
echo "Output:"
ls -dt runs/infer_mpc_blender/*_mpc_blender 2>/dev/null | head -1
