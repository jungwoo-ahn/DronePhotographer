#!/usr/bin/env bash
set -euo pipefail

# Parallel multi-GPU rendering with render_object.py.
# Launches one Blender process per GPU for maximum throughput.
#
# Usage:
#   bash scripts/render_object.sh
#
# Optional overrides:
#   BLENDER_BIN=blender/blender
#   SCENE_PATH=/abs/path/to/DogWalk.blend
#   OUTPUT_DIR=outputs
#   NUM_IMAGES=20
#   GPU_BACKEND=OPTIX
#   GPU_DEVICES="3 4 5"
#   BLENDER_THREADS=4

BLENDER_BIN="${BLENDER_BIN:-blender/blender}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
SCENE_PATH="${SCENE_PATH:-/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
NUM_IMAGES="${NUM_IMAGES:-20}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-3 4 5}"
RUN_NAME="${RUN_NAME:-smoke_rotation_fix}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

# Split GPU_DEVICES into an array
read -ra GPUS <<< "${GPU_DEVICES}"
NUM_WORKERS=${#GPUS[@]}

echo "Launching ${NUM_WORKERS} parallel workers..."
echo "  blender: ${BLENDER_BIN}"
echo "  scene:   ${SCENE_PATH}"
echo "  output:  ${OUTPUT_DIR}"
echo "  images:  ${NUM_IMAGES}"
echo "  backend: ${GPU_BACKEND}"
echo "  GPUs:    ${GPU_DEVICES} (1 worker per GPU)"

COMMON_ARGS=(
  --input_scene "${SCENE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --object_position -0.011 0.0364 0.8
  --num_images "${NUM_IMAGES}"
  --gpu_backend "${GPU_BACKEND}"
  --camera_radius_range 2 8
  --hemisphere
  --camera_direction_offsets 15 15 0
  --samples 32
  --adaptive_sampling --adaptive_threshold 0.02
  --max_bounces 2 --diffuse_bounces 1 --glossy_bounces 1 --transmission_bounces 1
  --persistent_data
  --blender_threads "${BLENDER_THREADS}"
  --num_workers "${NUM_WORKERS}"
)

LOG_DIR="${OUTPUT_DIR}/.render_logs"
mkdir -p "${LOG_DIR}"

PIDS=()
for i in "${!GPUS[@]}"; do
  gpu_id="${GPUS[$i]}"
  echo "  Starting worker ${i} on GPU ${gpu_id} (log: ${LOG_DIR}/worker_${i}.log)"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${BLENDER_BIN}" -b -t "${BLENDER_THREADS}" -P render_object.py -- \
    "${COMMON_ARGS[@]}" \
    --worker_index "${i}" \
    --gpu_devices 0 \
    > "${LOG_DIR}/worker_${i}.log" 2>&1 &
  PIDS+=($!)
done

# Monitor progress by counting rendered images
echo ""
while true; do
  # Check if any worker is still running
  STILL_RUNNING=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      STILL_RUNNING=1
      break
    fi
  done
  [ "${STILL_RUNNING}" -eq 0 ] && break

  # Find the output directory once it exists
  if [ -z "${IMAGES_DIR:-}" ]; then
    LATEST_DIR=$(ls -dt "${OUTPUT_DIR}"/*"${RUN_NAME}"* 2>/dev/null | head -1)
    if [ -n "${LATEST_DIR}" ] && [ -d "${LATEST_DIR}/images" ]; then
      IMAGES_DIR="${LATEST_DIR}/images"
    fi
  fi

  if [ -n "${IMAGES_DIR:-}" ]; then
    DONE=$(find "${IMAGES_DIR}" -name '*.png' 2>/dev/null | wc -l)
    printf "\r  Rendering: %d / %d images..." "${DONE}" "${NUM_IMAGES}"
  fi
  sleep 2
done

# Final count
if [ -n "${IMAGES_DIR:-}" ]; then
  DONE=$(find "${IMAGES_DIR}" -name '*.png' 2>/dev/null | wc -l)
  printf "\r  Rendering: %d / %d images... done.\n" "${DONE}" "${NUM_IMAGES}"
fi

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "Worker (pid ${pid}) failed. Check ${LOG_DIR}/" >&2
    FAIL=1
  fi
done

if [ "${FAIL}" -eq 1 ]; then
  echo "Some workers failed. Logs: ${LOG_DIR}/" >&2
  exit 1
fi

# Merge worker annotations into a single annotations.json
if [ "${NUM_WORKERS}" -gt 1 ]; then
  # Find the output directory (most recent matching run_name)
  LATEST_DIR=$(ls -dt "${OUTPUT_DIR}"/*"${RUN_NAME}"* 2>/dev/null | head -1)
  if [ -n "${LATEST_DIR}" ]; then
    python3 -c "
import json, sys
from pathlib import Path
run_dir = Path('${LATEST_DIR}')
all_annotations = []
for w in range(${NUM_WORKERS}):
    p = run_dir / f'annotations_worker{w}.json'
    if p.exists():
        all_annotations.extend(json.loads(p.read_text()))
all_annotations.sort(key=lambda x: x['image'])
with (run_dir / 'annotations.json').open('w') as f:
    json.dump(all_annotations, f, indent=2)
for w in range(${NUM_WORKERS}):
    p = run_dir / f'annotations_worker{w}.json'
    if p.exists():
        p.unlink()
print(f'Merged {len(all_annotations)} annotations into {run_dir / \"annotations.json\"}')
"
  fi
fi

echo "Render finished. ${NUM_IMAGES} images across ${NUM_WORKERS} GPUs."
