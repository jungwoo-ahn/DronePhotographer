#!/usr/bin/env bash
set -euo pipefail

# DronePhotographer v5.0 — multi-GPU placement-driven render driver.
# Wraps scripts/render_v5_from_placements.py.
#
# Usage:
#   bash render_v5.sh                 # full run (940 placements x NUM_IMAGES)
#   SMOKE=1 bash render_v5.sh         # smoke run (5 placements x 50 imgs)
#
# Optional overrides:
#   PLACEMENTS_DIR=data/vlm_object_placing
#   ASSETS_ROOT=.
#   OUTPUT_DIR=outputs
#   RUN_NAME=v5_full_run
#   NUM_IMAGES=2000
#   GPU_BACKEND=OPTIX
#   GPU_DEVICES="1 2 3 4 5"
#   BLENDER_THREADS=4
#   BLENDER_BIN=blender/blender
#   SAMPLES=64

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

PLACEMENTS_DIR="${PLACEMENTS_DIR:-${REPO_ROOT}/data/vlm_object_placing}"
ASSETS_ROOT="${ASSETS_ROOT:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
NUM_IMAGES="${NUM_IMAGES:-2000}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-1 2 3 4 5}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
SAMPLES="${SAMPLES:-64}"

if [ "${SMOKE:-0}" -eq 1 ]; then
  RUN_NAME="${RUN_NAME:-v5_smoke}"
  MAX_PLACEMENTS=30
  NUM_IMAGES=5
else
  TIMESTAMP=$(date +%y%m%d_%H%M%S)
  RUN_NAME="${RUN_NAME:-v5_full_${TIMESTAMP}}"
  MAX_PLACEMENTS=""
fi

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

ARGS=(
  --placements_dir "${PLACEMENTS_DIR}"
  --assets_root "${ASSETS_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --num_images_per_placement "${NUM_IMAGES}"
  --gpu_devices ${GPU_DEVICES}
  --gpu_backend "${GPU_BACKEND}"
  --blender_bin "${BLENDER_BIN}"
  --blender_threads "${BLENDER_THREADS}"
  --samples "${SAMPLES}"
  --max_bounces 4
  --diffuse_bounces 2
  --glossy_bounces 2
  --transmission_bounces 2
  --persistent_data
  --camera_radius_range 2 8
  --hemisphere
  --camera_direction_offsets 15 15 0
  --use_aabb_center
)

if [ -n "${MAX_PLACEMENTS}" ]; then
  ARGS+=(--max_placements "${MAX_PLACEMENTS}")
fi
if [ "${RESUME:-0}" -eq 1 ]; then
  ARGS+=(--resume)
fi

echo "DronePhotographer v5.0"
echo "  placements_dir: ${PLACEMENTS_DIR}"
echo "  output:         ${OUTPUT_DIR}/${RUN_NAME}"
echo "  num_images:     ${NUM_IMAGES} per placement"
echo "  GPUs:           ${GPU_DEVICES}"
[ -n "${MAX_PLACEMENTS}" ] && echo "  max_placements: ${MAX_PLACEMENTS} (smoke)"

python3 "${REPO_ROOT}/scripts/render_v5_from_placements.py" "${ARGS[@]}"
