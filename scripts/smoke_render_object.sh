#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test for render_object.py after rotation changes.
# Usage:
#   bash scripts/smoke_render_object.sh
#
# Optional overrides:
#   BLENDER_BIN=blender/blender
#   SCENE_PATH=/abs/path/to/DogWalk.blend
#   OUTPUT_DIR=outputs
#   NUM_IMAGES=20
#   GPU_BACKEND=OPTIX
#   GPU_DEVICES="6 7"

BLENDER_BIN="${BLENDER_BIN:-blender/blender}"
SCENE_PATH="${SCENE_PATH:-/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
NUM_IMAGES="${NUM_IMAGES:-20}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-6 7}"

echo "Running render smoke test..."
echo "  blender: ${BLENDER_BIN}"
echo "  scene:   ${SCENE_PATH}"
echo "  output:  ${OUTPUT_DIR}"
echo "  images:  ${NUM_IMAGES}"
echo "  backend: ${GPU_BACKEND}"
echo "  devices: ${GPU_DEVICES}"

"${BLENDER_BIN}" -b -P render_object.py -- \
  --input_scene "${SCENE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name smoke_rotation_fix \
  --object_position -0.011 0.0364 0.8 \
  --num_images "${NUM_IMAGES}" \
  --gpu_backend "${GPU_BACKEND}" \
  --camera_radius_range 2 8 \
  --hemisphere \
  --camera_direction_offsets 15 15 0 \
  --samples 32 \
  --adaptive_sampling --adaptive_threshold 0.02 \
  --max_bounces 2 --diffuse_bounces 1 --glossy_bounces 1 --transmission_bounces 1 \
  --persistent_data \
  --gpu_devices ${GPU_DEVICES}

echo "Smoke render finished."
