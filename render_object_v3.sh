#!/usr/bin/env bash
set -euo pipefail

# Parallel multi-GPU rendering with render_object_v3.py.
# Launches one Blender process per GPU for maximum throughput.
#
# Usage:
#   bash render_object_v3.sh
#
# Optional overrides:
#   BLENDER_BIN=blender/blender
#   SCENE_PATH=/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend
#   OUTPUT_DIR=outputs
#   RUN_NAME=full_10000
#   NUM_IMAGES=8000
#   GPU_BACKEND=OPTIX
#   GPU_DEVICES="3 4 5"
#   BLENDER_THREADS=4

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
SCENE_PATH="${SCENE_PATH:-/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-v3_8k}"
NUM_IMAGES="${NUM_IMAGES:-8000}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-3 4 5}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

read -ra GPUS <<< "${GPU_DEVICES}"
NUM_WORKERS=${#GPUS[@]}

# Kill all child processes on exit (Ctrl+C, error, etc.)
cleanup() {
  echo ""
  echo "Stopping all workers..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Generate a single run directory for all workers
TIMESTAMP=$(date +%y%m%d_%H%M%S)
SCENE_STEM=$(basename "${SCENE_PATH}" .blend)
RUN_DIR="${OUTPUT_DIR}/${SCENE_STEM}_${RUN_NAME}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}/images"

START_TIME=$(date +%s)

echo "Launching ${NUM_WORKERS} parallel workers..."
echo "  blender:  ${BLENDER_BIN}"
echo "  scene:    ${SCENE_PATH}"
echo "  output:   ${OUTPUT_DIR}"
echo "  run_name: ${RUN_NAME}"
echo "  images:   ${NUM_IMAGES}"
echo "  backend:  ${GPU_BACKEND}"
echo "  GPUs:     ${GPU_DEVICES} (1 worker per GPU)"

COMMON_ARGS=(
  --input_scene "${SCENE_PATH}"
  --output_run_dir "${RUN_DIR}"
  --object_name RIG-Snowman.001
  --num_images "${NUM_IMAGES}"
  --gpu_backend "${GPU_BACKEND}"
  --camera_radius_range 1 8
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
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${BLENDER_BIN}" -b -t "${BLENDER_THREADS}" -P "${REPO_ROOT}/render_object_v3.py" -- \
    "${COMMON_ARGS[@]}" \
    --worker_index "${i}" \
    --gpu_devices 0 \
    > "${LOG_DIR}/worker_${i}.log" 2>&1 &
  PIDS+=($!)
done

# Monitor progress by counting rendered images
echo ""
IMAGES_DIR="${RUN_DIR}/images"
while true; do
  STILL_RUNNING=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      STILL_RUNNING=1
      break
    fi
  done
  [ "${STILL_RUNNING}" -eq 0 ] && break

  DONE=$(find "${IMAGES_DIR}" -name '*.png' 2>/dev/null | wc -l)
  ELAPSED=$(( $(date +%s) - START_TIME ))
  MINS=$(( ELAPSED / 60 ))
  SECS=$(( ELAPSED % 60 ))
  printf "\r  Rendering: %d / %d images... [%dm %ds]" "${DONE}" "${NUM_IMAGES}" "${MINS}" "${SECS}"
  sleep 2
done

DONE=$(find "${IMAGES_DIR}" -name '*.png' 2>/dev/null | wc -l)
ELAPSED=$(( $(date +%s) - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))
printf "\r  Rendering: %d / %d images... done. [%dm %ds]\n" "${DONE}" "${NUM_IMAGES}" "${MINS}" "${SECS}"

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
    python3 -c "
import json
from pathlib import Path
run_dir = Path('${RUN_DIR}')
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

# Disable cleanup trap on successful exit
trap - EXIT
echo "Render finished. ${NUM_IMAGES} images across ${NUM_WORKERS} GPUs."
