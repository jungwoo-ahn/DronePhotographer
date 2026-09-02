#!/usr/bin/env bash
set -euo pipefail

# DronePhotographer v5.0 — 8x RTX 3090 (24GB VRAM) optimized profile.
#
# v5.1 (2026-04-27): switched to 1-GPU-per-placement parallelism.
#   Previous scheme had 8 GPUs sync the SAME .blend per placement (7/8
#   redundant CPU sync work). New scheme launches 8 Python processes,
#   each pinned to one GPU, each handling a 1/8 slice of placements.
#   Sync cost is paid once per scene per GPU instead of 8x.
#
# Per worker: one Blender process renders ALL frames of its assigned
# placements sequentially on a single GPU. Persistent_data keeps the
# synced scene + BVH alive across frames within a placement.
#
# Usage:
#   bash render_v5_3090x8.sh                  # full run (940 placements)
#   SMOKE=1 bash render_v5_3090x8.sh          # smoke (5 placements x 50 imgs)
#   RESUME=1 bash render_v5_3090x8.sh         # skip placements with valid annotations.json
#
# Tunable env vars (defaults shown):
#   GPU_DEVICES="0 1 2 3 4 5 6 7"
#   BLENDER_THREADS=4                          # CPU threads per Blender (8 procs x 4 = 32)
#   NUM_IMAGES=2000                            # images per placement
#   SAMPLES=32                                 # max samples (adaptive cuts where it can)
#   ADAPTIVE=1                                 # 1=adaptive sampling on, 0=fixed samples
#   ADAPTIVE_THRESHOLD=0.02
#   GPU_BACKEND=OPTIX                          # RTX 3090 has RT cores -> OPTIX is fastest
#   MAX_BOUNCES=3 / DIFFUSE_BOUNCES=1 / GLOSSY_BOUNCES=2 / TRANSMISSION_BOUNCES=2
#   PLACEMENTS_DIR / ASSETS_ROOT / OUTPUT_DIR / RUN_NAME / BLENDER_BIN

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

PLACEMENTS_DIR="${PLACEMENTS_DIR:-${REPO_ROOT}/data/vlm_object_placing}"
ASSETS_ROOT="${ASSETS_ROOT:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
NUM_IMAGES="${NUM_IMAGES:-2000}"
GPU_BACKEND="${GPU_BACKEND:-OPTIX}"
GPU_DEVICES="${GPU_DEVICES:-0 1 2 3 4 5 6 7}"
BLENDER_THREADS="${BLENDER_THREADS:-4}"
BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
SAMPLES="${SAMPLES:-32}"
ADAPTIVE="${ADAPTIVE:-1}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-0.02}"
MAX_BOUNCES="${MAX_BOUNCES:-3}"
DIFFUSE_BOUNCES="${DIFFUSE_BOUNCES:-1}"
GLOSSY_BOUNCES="${GLOSSY_BOUNCES:-2}"
TRANSMISSION_BOUNCES="${TRANSMISSION_BOUNCES:-2}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"   # Blender processes per GPU. GPU is mostly
                                       # idle during scene sync (CPU-bound), so
                                       # 2 procs/GPU lets one render while the
                                       # other syncs. Diminishing returns at 3+.

if [ "${SMOKE:-0}" -eq 1 ]; then
  RUN_NAME="${RUN_NAME:-v5_smoke_3090x8}"
  MAX_PLACEMENTS=5
  NUM_IMAGES=50
else
  TIMESTAMP=$(date +%y%m%d_%H%M%S)
  RUN_NAME="${RUN_NAME:-v5_3090x8_${TIMESTAMP}}"
  MAX_PLACEMENTS=""
fi

# Bound per-process BLAS threads so 8 Blenders x N threads doesn't oversubscribe.
export OMP_NUM_THREADS="${BLENDER_THREADS}"
export OPENBLAS_NUM_THREADS="${BLENDER_THREADS}"
export MKL_NUM_THREADS="${BLENDER_THREADS}"

# Common args (per-GPU args added in the launch loop)
COMMON_ARGS=(
  --placements_dir "${PLACEMENTS_DIR}"
  --assets_root "${ASSETS_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --num_images_per_placement "${NUM_IMAGES}"
  --gpu_backend "${GPU_BACKEND}"
  --blender_bin "${BLENDER_BIN}"
  --blender_threads "${BLENDER_THREADS}"
  --samples "${SAMPLES}"
  --max_bounces "${MAX_BOUNCES}"
  --diffuse_bounces "${DIFFUSE_BOUNCES}"
  --glossy_bounces "${GLOSSY_BOUNCES}"
  --transmission_bounces "${TRANSMISSION_BOUNCES}"
  --persistent_data
  --camera_radius_range 2 8
  --hemisphere
  --camera_direction_offsets 15 15 0
  --use_aabb_center
)

if [ "${ADAPTIVE}" -eq 1 ]; then
  COMMON_ARGS+=(--adaptive_sampling --adaptive_threshold "${ADAPTIVE_THRESHOLD}")
fi
if [ -n "${MAX_PLACEMENTS}" ]; then
  COMMON_ARGS+=(--max_placements "${MAX_PLACEMENTS}")
fi
if [ "${RESUME:-0}" -eq 1 ]; then
  COMMON_ARGS+=(--resume)
fi

NUM_GPUS=$(echo ${GPU_DEVICES} | wc -w)
TOTAL_SLICES=$(( NUM_GPUS * PROCS_PER_GPU ))
TOTAL_THREADS=$(( BLENDER_THREADS * TOTAL_SLICES ))

echo "DronePhotographer v5.2 — 8x RTX 3090 profile (${PROCS_PER_GPU} procs/GPU)"
echo "  placements_dir:  ${PLACEMENTS_DIR}"
echo "  output:          ${OUTPUT_DIR}/${RUN_NAME}"
echo "  num_images:      ${NUM_IMAGES} per placement"
echo "  GPUs:            ${GPU_DEVICES}  (${NUM_GPUS} devices, backend=${GPU_BACKEND})"
echo "  procs/gpu:       ${PROCS_PER_GPU}  (total ${TOTAL_SLICES} slices)"
echo "  threads/blender: ${BLENDER_THREADS}  (total ${TOTAL_THREADS} CPU threads)"
echo "  samples:         ${SAMPLES}  adaptive=${ADAPTIVE} (threshold=${ADAPTIVE_THRESHOLD})"
echo "  bounces:         max=${MAX_BOUNCES} diff=${DIFFUSE_BOUNCES} gloss=${GLOSSY_BOUNCES} trans=${TRANSMISSION_BOUNCES}"
[ -n "${MAX_PLACEMENTS}" ] && echo "  max_placements:  ${MAX_PLACEMENTS} (SMOKE)"
[ "${RESUME:-0}" -eq 1 ] && echo "  resume:          on"
echo

# Make sure master dir exists before slice logs are written into it
mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"

# Launch PROCS_PER_GPU Python orchestrators per GPU; each handles 1/TOTAL_SLICES slice
PIDS=()
SLICE_IDX=0
for GPU in ${GPU_DEVICES}; do
  for proc_n in $(seq 1 ${PROCS_PER_GPU}); do
    LOG="${OUTPUT_DIR}/${RUN_NAME}/.slice_${SLICE_IDX}.log"
    echo "  launching slice ${SLICE_IDX}/${TOTAL_SLICES} on GPU ${GPU} (proc ${proc_n}/${PROCS_PER_GPU})  -> ${LOG}"
    python3 "${REPO_ROOT}/scripts/render_v5_from_placements.py" \
      "${COMMON_ARGS[@]}" \
      --gpu_devices ${GPU} \
      --placement_slice "${SLICE_IDX}:${TOTAL_SLICES}" \
      > "${LOG}" 2>&1 &
    PIDS+=($!)
    SLICE_IDX=$((SLICE_IDX + 1))
  done
done

echo
echo "All ${TOTAL_SLICES} workers launched. Waiting for completion..."

EXIT=0
SLICE_IDX=0
for PID in "${PIDS[@]}"; do
  if wait $PID; then
    echo "  slice ${SLICE_IDX} (pid=${PID}) -> ok"
  else
    RC=$?
    echo "  slice ${SLICE_IDX} (pid=${PID}) -> FAILED (rc=${RC})"
    EXIT=$RC
  fi
  SLICE_IDX=$((SLICE_IDX + 1))
done

echo
echo "All slices done. exit=${EXIT}"
exit $EXIT
