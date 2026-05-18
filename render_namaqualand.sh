#!/usr/bin/env bash
# Render rp_posedplus character in Namaqualand scene using render_object_v3.
# Usage: bash render_namaqualand.sh

export SCENE_PATH="/home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/namaqualand/Namaqualand.blend"
export RUN_NAME="namaqualand_v3"
export NUM_IMAGES="${NUM_IMAGES:-10000}"
export GPU_DEVICES="${GPU_DEVICES:-1 2 3 4 5 6 7}"

# Override COMMON_ARGS in render_object_v3.sh by patching inline
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"

export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

read -ra GPUS <<< "${GPU_DEVICES}"
NUM_WORKERS=${#GPUS[@]}

set -euo pipefail

cleanup() {
  echo ""
  echo "Stopping all workers..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

TIMESTAMP=$(date +%y%m%d_%H%M%S)
SCENE_STEM=$(basename "${SCENE_PATH}" .blend)
RUN_DIR="${OUTPUT_DIR}/${SCENE_STEM}_${RUN_NAME}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}/images"

START_TIME=$(date +%s)

echo "=== Namaqualand Render ==="
echo "  blender:  ${BLENDER_BIN}"
echo "  scene:    ${SCENE_PATH}"
echo "  object:   rp_posedplus_00068_18_100k"
echo "  position: (16.117, -5.1769, 5.6461)"
echo "  rotation: 135 deg"
echo "  output:   ${RUN_DIR}"
echo "  images:   ${NUM_IMAGES}"
echo "  GPUs:     ${GPU_DEVICES} (${NUM_WORKERS} workers)"

COMMON_ARGS=(
  --input_scene "${SCENE_PATH}"
  --output_run_dir "${RUN_DIR}"
  --input_object /home/nas5/jungwooahn/datasets/DronePhotos/assets/objects/rp_posedplus_00068_18_100k
  --object_position 16.117 -5.1769 5.6461
  --rotation_z_deg 135
  --scale 0.01004
  --use_aabb_center
  --sky_strength 0.2
  --num_images "${NUM_IMAGES}"
  --gpu_backend "${GPU_BACKEND}"
  --camera_radius_range 0.5 6
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

# Merge worker annotations
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

trap - EXIT
echo "Render finished. ${NUM_IMAGES} images across ${NUM_WORKERS} GPUs."
echo "Output: ${RUN_DIR}"
