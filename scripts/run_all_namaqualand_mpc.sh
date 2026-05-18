#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_all_namaqualand_mpc.sh [GPU_INDEX] [SCRIPT_NAMES...]
# Examples:
#   bash scripts/run_all_namaqualand_mpc.sh 0                    # all 6 on GPU 0
#   bash scripts/run_all_namaqualand_mpc.sh 0 behind_aerial front_eyelevel low_heroic
#   bash scripts/run_all_namaqualand_mpc.sh 1 topdown_birdeye upper_right left_side

GPU_INDEX="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
shift || true
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

# Score weights: bbox + camera_to_object (targets now corrected for obj_fwd convention)
export SCORE_WEIGHTS_JSON='{"bbox_occupancy_ratio":2.0,"bbox_margin_top":1.0,"bbox_margin_bottom":1.0,"bbox_margin_left":1.0,"bbox_margin_right":1.0,"bbox_aspect_ratio":1.0,"bbox_centroid_offset":2.0,"camera_to_object_fx":1.0,"camera_to_object_fy":1.0,"camera_to_object_fz":1.0,"camera_to_object_ux":1.0,"camera_to_object_uy":1.0,"camera_to_object_uz":1.0}'

ALL_SCRIPTS=(
  behind_aerial
  front_eyelevel
  low_heroic
  topdown_birdeye
  upper_right
  left_side
)

if [ $# -gt 0 ]; then
  SCRIPTS=("$@")
else
  SCRIPTS=("${ALL_SCRIPTS[@]}")
fi

LOG_DIR="/tmp/mpc_namaqualand_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Logs: ${LOG_DIR}"
echo "Launching ${#SCRIPTS[@]} scripts..."
echo ""

PIDS=()
for script in "${SCRIPTS[@]}"; do
  log_file="${LOG_DIR}/${script}.log"
  echo "[$(date +%H:%M:%S)] Starting ${script} → ${log_file}"
  bash "scripts/infer_mpc_namaqualand_${script}.sh" > "${log_file}" 2>&1 &
  PIDS+=($!)
  # 2-second gap to avoid output directory timestamp collision
  sleep 30
done

echo ""
echo "All launched. PIDs: ${PIDS[*]}"
echo ""

# Wait and report
FAILED=0
for i in "${!SCRIPTS[@]}"; do
  script="${SCRIPTS[$i]}"
  pid="${PIDS[$i]}"
  if wait "${pid}"; then
    echo "[$(date +%H:%M:%S)] DONE  ${script} (PID ${pid})"
  else
    echo "[$(date +%H:%M:%S)] FAIL  ${script} (PID ${pid}, exit=$?)"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [ "${FAILED}" -eq 0 ]; then
  echo "All ${#SCRIPTS[@]} runs completed successfully."
else
  echo "${FAILED}/${#SCRIPTS[@]} runs failed. Check logs in ${LOG_DIR}"
fi

# Print output directories
echo ""
echo "Output directories:"
ls -dt runs/infer_mpc_blender/*_mpc_blender 2>/dev/null | head -"${#SCRIPTS[@]}"
