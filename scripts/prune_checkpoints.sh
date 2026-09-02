#!/usr/bin/env bash
# Disk guard for the big VLA runs (GR00T/pi0.5 checkpoints are 24-40GB each, and lerobot
# has no built-in checkpoint rotation). Keeps the KEEP newest numbered checkpoints per run
# (newest = resume point; a few prior = recon-later buffer) and deletes older ones.
# No GPU. Run in background; stop with TaskStop.
set -u
KEEP="${KEEP:-4}"
RUNS=("runs/groot_drone" "runs/pi05_drone")
cd /home/nas5/jooyeolyun/repos/DronePhotographer
while true; do
  for r in "${RUNS[@]}"; do
    d="$r/checkpoints"
    [ -d "$d" ] || continue
    # numbered checkpoint dirs, newest last; delete all but the last $KEEP
    mapfile -t cks < <(ls -d "$d"/[0-9]*/ 2>/dev/null | sort)
    n=${#cks[@]}
    if [ "$n" -gt "$KEEP" ]; then
      for ((i=0; i<n-KEEP; i++)); do
        rm -rf "${cks[$i]}" && echo "[$(date +%H:%M)] pruned ${cks[$i]}"
      done
    fi
  done
  sleep 120
done
