#!/usr/bin/env bash
# Wait for a sustained-free GPU on a shared node, then launch a training run on it
# (detached, so it survives the launching shell). A GPU counts as free only if it
# stays under the memory threshold across two checks spaced CONFIRM_S apart — so a
# neighbour's momentary dip doesn't make us collide with them.
#
# Usage:
#   scripts/wait_for_gpu_and_train.sh <config.yaml> [extra train args...]
# Env:
#   PYTHON          python interpreter (default: the drone conda env)
#   CANDIDATES      space-separated GPU indices to consider (default: all)
#   FREE_MB         max used-MB to call a GPU free (default 1000)
#   POLL_S          seconds between scans (default 60)
#   CONFIRM_S       seconds the GPU must stay free before we claim it (default 45)
set -u

CONFIG="${1:?usage: wait_for_gpu_and_train.sh <config.yaml> [train args...]}"; shift || true
PYTHON="${PYTHON:-/home/jooyeolyun/anaconda3/envs/drone/bin/python}"
FREE_MB="${FREE_MB:-1000}"
POLL_S="${POLL_S:-60}"
CONFIRM_S="${CONFIRM_S:-45}"
REPO="/home/nas5/jooyeolyun/repos/DronePhotographer"
cd "$REPO"
mkdir -p logs

free_gpus() {  # echo indices currently under the memory threshold
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v t="$FREE_MB" '$2+0 < t {gsub(/ /,"",$1); print $1}'
  if [ -n "${CANDIDATES:-}" ]; then :; fi
}
in_candidates() { [ -z "${CANDIDATES:-}" ] && return 0; for c in $CANDIDATES; do [ "$c" = "$1" ] && return 0; done; return 1; }

echo "$(date '+%F %T') waiting for a free GPU (free<${FREE_MB}MB, candidates='${CANDIDATES:-all}') for: $CONFIG"
while true; do
  picked=""
  for g in $(free_gpus); do
    in_candidates "$g" || continue
    sleep "$CONFIRM_S"
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$FREE_MB" ]; then picked="$g"; break; fi
  done
  if [ -n "$picked" ]; then
    ts=$(date '+%Y%m%d_%H%M%S')
    log="logs/train_gpu${picked}_${ts}.log"
    echo "$(date '+%F %T') claiming GPU $picked -> $log"
    CUDA_VISIBLE_DEVICES="$picked" PYTHONPATH=. nohup "$PYTHON" scripts/train_vla_policy.py \
      --config "$CONFIG" "$@" > "$log" 2>&1 &
    echo "$(date '+%F %T') launched pid $! on GPU $picked"
    echo "$log"
    exit 0
  fi
  sleep "$POLL_S"
done
