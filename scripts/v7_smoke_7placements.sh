#!/usr/bin/env bash
# Run v7 pair-sampling smoke on 7 diverse placements.
# 4 frames per clip (endpoints always included) via --render-num-frames 4.
# Uses GPU 7 (must be free).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BLENDER="${REPO_ROOT}/blender/blender"
OUT_DIR="${REPO_ROOT}/outputs/v7_pair_smoke_7run"
LOG_DIR="${OUT_DIR}/_logs"
mkdir -p "$LOG_DIR"

PLACEMENTS=(
  "data/vlm_object_placing_v6_260428_061326/Abandoned-alley_9ee2b453__Bald-Worker-Hands-on-His-Hips_7d114532.json"
  "data/vlm_object_placing_v6_260428_061326/Forest-field_3c5ba348__Casual-Crouch_b3e2a7cc.json"
  "data/vlm_object_placing_v6_260428_061326/cafe-interior-lo_7e263422-a9e9-4aae-96a2-4e4d6c589f68__Bar-Connoisseur_bd5a51d0.json"
  "data/vlm_object_placing_v6_260428_061326/Apartment-garden_fcd5b3f7__Elegant-Woman_33974376.json"
  "data/vlm_object_placing_v6_260428_061326/Modern-Art-Gallery_770933c8__Casual-man-standing_89a477f1.json"
  "data/vlm_object_placing_v6_260428_061326/Gas-Station_b8467c86__andrew_16e03125-959b-4312-8e8a-a7cb23bf3a1c.json"
  "data/vlm_object_placing_v6_260428_061326/Cozy-Cabin_955470a2__Buddy-Sitting_6de152bb.json"
)

GPU_INDEX="${GPU_INDEX:-7}"
SEED="${SEED:-0}"
RENDER_NUM_FRAMES="${RENDER_NUM_FRAMES:-4}"
RENDER_SAMPLES="${RENDER_SAMPLES:-32}"

START_TS=$(date +%s)
echo "[runner] gpu=$GPU_INDEX seed=$SEED num_frames=$RENDER_NUM_FRAMES samples=$RENDER_SAMPLES"
echo "[runner] out_dir=$OUT_DIR"
echo "[runner] $(date)"

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
    echo "[ok] ${dt}s — log: ${log}"
    grep -E "^\[smoke\] (placement=|rendered|dir=)" "$log" | tail -5 || true
  else
    echo "[FAIL rc=$rc] ${dt}s — see ${log}"
    tail -20 "$log" || true
  fi
done

END_TS=$(date +%s)
echo
echo "[runner] total $((END_TS - START_TS))s"
echo "[runner] reports:"
find "$OUT_DIR" -maxdepth 3 -name '*.html' | sort
