#!/usr/bin/env bash
set -euo pipefail

# V7 Stage 2 — multi-GPU launcher.
#
# Spawns one Blender process per GPU; each takes a 1/N slice of this side's
# placement list. Persistent_data lets consecutive same-scene placements
# share scene loads within a slice.
#
# Tunable env vars (defaults shown):
#   GPU_DEVICES="0 1 2 3 4 5 6"
#   PROCS_PER_GPU=1                      # Stage 2 is render-bound; 1 proc/GPU is optimal
#   ASSIGNMENT_FILE=splits/v7_stage2_assignments.json
#   SIDE=jungwooahn                      # which half this machine renders
#   STAGE1_DIR=outputs/v7_stage1_sample
#   PLACEMENTS_V6_DIR=data/vlm_object_placing_v6_260428_061326
#   ASSETS_ROOT=<auto>                   # auto-detects ../DronePhotographer if exists
#   OUT_DIR=outputs/v7_stage2_renders
#   FRAMES_PER_PAIR=32
#   RENDER_SAMPLES=32
#   RESOLUTION="1024 768"
#   BLENDER_BIN=blender/blender          # symlink from main repo if missing
#   RESUME=1
#   PILOT_COUNT=0                        # >0 for end-to-end smoke

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

GPU_DEVICES="${GPU_DEVICES:-0 1 2 3 4 5 6}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
ASSIGNMENT_FILE="${ASSIGNMENT_FILE:-splits/v7_stage2_assignments.json}"
SIDE="${SIDE:-jungwooahn}"
STAGE1_DIR="${STAGE1_DIR:-outputs/v7_stage1_sample}"
PLACEMENTS_V6_DIR="${PLACEMENTS_V6_DIR:-data/vlm_object_placing_v6_260428_061326}"

# Default assets-root: main DronePhotographer repo if it's a sibling, else self.
if [ -z "${ASSETS_ROOT:-}" ]; then
  if [ -d "${REPO_ROOT}/../DronePhotographer/data/scenes" ]; then
    ASSETS_ROOT="$(cd "${REPO_ROOT}/../DronePhotographer" && pwd)"
  else
    ASSETS_ROOT="${REPO_ROOT}"
  fi
fi

OUT_DIR="${OUT_DIR:-outputs/v7_stage2_renders}"
FRAMES_PER_PAIR="${FRAMES_PER_PAIR:-32}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"
RESOLUTION="${RESOLUTION:-1024 768}"
BLENDER_BIN="${BLENDER_BIN:-${REPO_ROOT}/blender/blender}"
RESUME="${RESUME:-1}"
PILOT_COUNT="${PILOT_COUNT:-0}"

NUM_GPUS=$(echo ${GPU_DEVICES} | wc -w)
TOTAL_SLICES=$(( NUM_GPUS * PROCS_PER_GPU ))

LOG_DIR="${REPO_ROOT}/${OUT_DIR}/_logs"
mkdir -p "${LOG_DIR}"

echo "v7 Stage 2 launcher"
echo "  repo:            ${REPO_ROOT}"
echo "  assets-root:     ${ASSETS_ROOT}"
echo "  blender:         ${BLENDER_BIN}"
echo "  assignment:      ${ASSIGNMENT_FILE} (side=${SIDE})"
echo "  stage1-dir:      ${STAGE1_DIR}"
echo "  v6 placements:   ${PLACEMENTS_V6_DIR}"
echo "  out:             ${OUT_DIR}"
echo "  GPUs:            ${GPU_DEVICES}  (${NUM_GPUS} devices)"
echo "  procs/gpu:       ${PROCS_PER_GPU}  (total ${TOTAL_SLICES} slices)"
echo "  frames/pair:     ${FRAMES_PER_PAIR}"
echo "  render-samples:  ${RENDER_SAMPLES}"
echo "  resolution:      ${RESOLUTION}"
echo "  resume:          ${RESUME}"
echo "  pilot-count:     ${PILOT_COUNT}  (0 = full)"
echo "  logs:            ${LOG_DIR}"

if [ ! -x "${BLENDER_BIN}" ]; then
  echo "ERROR: blender binary not executable at ${BLENDER_BIN}" >&2
  echo "Hint: ln -s ../DronePhotographer/blender ${REPO_ROOT}/blender" >&2
  exit 2
fi
if [ ! -f "${REPO_ROOT}/${ASSIGNMENT_FILE}" ]; then
  echo "ERROR: assignment file missing: ${REPO_ROOT}/${ASSIGNMENT_FILE}" >&2
  exit 2
fi

COMMON_ARGS=(
  --stage1-dir "${STAGE1_DIR}"
  --placements-v6-dir "${PLACEMENTS_V6_DIR}"
  --assets-root "${ASSETS_ROOT}"
  --out-dir "${OUT_DIR}"
  --assignment-file "${ASSIGNMENT_FILE}"
  --side "${SIDE}"
  --slice-count "${TOTAL_SLICES}"
  --frames-per-pair "${FRAMES_PER_PAIR}"
  --render-samples "${RENDER_SAMPLES}"
  --resolution ${RESOLUTION}
)
if [ "${RESUME}" -eq 1 ]; then
  COMMON_ARGS+=(--resume)
fi
if [ "${PILOT_COUNT}" -gt 0 ]; then
  COMMON_ARGS+=(--pilot-count "${PILOT_COUNT}")
fi

PIDS=()
SLICE_IDX=0
for GPU in ${GPU_DEVICES}; do
  for _ in $(seq 1 ${PROCS_PER_GPU}); do
    LOG="${LOG_DIR}/slice_${SLICE_IDX}.log"
    echo "  launching slice ${SLICE_IDX}/${TOTAL_SLICES} on GPU ${GPU}  -> ${LOG}"
    "${BLENDER_BIN}" -b -P "${REPO_ROOT}/scripts/v7_stage2_render.py" -- \
      "${COMMON_ARGS[@]}" \
      --slice-index "${SLICE_IDX}" \
      --gpu-index "${GPU}" \
      > "${LOG}" 2>&1 &
    PIDS+=($!)
    SLICE_IDX=$((SLICE_IDX + 1))
  done
done

echo
echo "All ${TOTAL_SLICES} workers launched. Waiting..."

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
