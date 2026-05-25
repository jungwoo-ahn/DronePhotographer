#!/usr/bin/env bash
# Re-run the 3 placements that failed in the first sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BLENDER="${REPO_ROOT}/blender/blender"
OUT_DIR="${REPO_ROOT}/outputs/v7_pair_smoke_7run"
LOG_DIR="${OUT_DIR}/_logs"
mkdir -p "$LOG_DIR"

PLACEMENTS=(
  "data/vlm_object_placing_v6_260428_061326/Forest-field_3c5ba348__standing-girl-ch_cdef5686-8f83-407c-80b0-5947093216fd.json"
  "data/vlm_object_placing_v6_260428_061326/Beach-bar_2c517ab3__Young-Tall-Guy-Talking-by-Phone_7e126859.json"
  "data/vlm_object_placing_v6_260428_061326/Cozy-Cabin_955470a2__Summer-Walk_3a48a916.json"
)

GPU_INDEX="${GPU_INDEX:-7}"
SEED="${SEED:-0}"
RENDER_NUM_FRAMES="${RENDER_NUM_FRAMES:-4}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"

START_TS=$(date +%s)
echo "[runner] replacement sweep · gpu=$GPU_INDEX seed=$SEED num_frames=$RENDER_NUM_FRAMES"

for i in "${!PLACEMENTS[@]}"; do
  pj="${PLACEMENTS[$i]}"
  name=$(basename "$pj" .json)
  log="$LOG_DIR/${name}.log"
  echo
  echo "==[ $((i+1))/${#PLACEMENTS[@]} ] $name =="
  t0=$(date +%s)
  set +e
  "$BLENDER" -b -P scripts/v7_sample_pairs_smoke.py -- \
      --placement-json "$pj" \
      --seed "$SEED" \
      --out-dir "$OUT_DIR" \
      --render-num-frames "$RENDER_NUM_FRAMES" \
      --render-samples "$RENDER_SAMPLES" \
      --gpu-index "$GPU_INDEX" \
      > "$log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)
  dt=$((t1 - t0))
  if [ $rc -eq 0 ]; then
    echo "[ok] ${dt}s"
    grep -E "^\[smoke\] (placement=|accepted=|rendered)" "$log" | tail -3 || true
  else
    echo "[FAIL rc=$rc] ${dt}s — see ${log}"
    tail -15 "$log" || true
  fi
done

END_TS=$(date +%s)
echo
echo "[runner] total $((END_TS - START_TS))s"
