#!/usr/bin/env bash
set -euo pipefail

# Full render launcher (non-test).
# Defaults match your requested setup: 10,000 images on DogWalk with OPTIX.
#
# Usage:
#   bash render_object.sh
#
# Optional overrides:
#   BLENDER_BIN=blender/blender
#   SCENE_PATH=/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend
#   OUTPUT_DIR=outputs
#   RUN_NAME=full_10000
#   NUM_IMAGES=8000
#   GPU_BACKEND=OPTIX
#   GPU_DEVICES="3 4 5"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
SCENE_PATH="${SCENE_PATH:-/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-v2_10k}"
NUM_IMAGES="${NUM_IMAGES:-8000}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-3 4 5}"

read -r -a GPU_DEVICE_ARRAY <<< "${GPU_DEVICES}"

echo "Starting full render run..."
echo "  blender:  ${BLENDER_BIN}"
echo "  scene:    ${SCENE_PATH}"
echo "  output:   ${OUTPUT_DIR}"
echo "  run_name: ${RUN_NAME}"
echo "  images:   ${NUM_IMAGES}"
echo "  backend:  ${GPU_BACKEND}"
echo "  devices:  ${GPU_DEVICES}"

"${BLENDER_BIN}" -b -P "${REPO_ROOT}/render_object.py" -- \
  --input_scene "${SCENE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --object_position -0.011 0.0364 0.8 \
  --num_images "${NUM_IMAGES}" \
  --gpu_backend "${GPU_BACKEND}" \
  --camera_radius_range 1 8 \
  --hemisphere \
  --camera_direction_offsets 15 15 0 \
  --samples 32 \
  --adaptive_sampling --adaptive_threshold 0.02 \
  --max_bounces 2 --diffuse_bounces 1 --glossy_bounces 1 --transmission_bounces 1 \
  --persistent_data \
  --gpu_devices "${GPU_DEVICE_ARRAY[@]}"

echo "Render complete."
