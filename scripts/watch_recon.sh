#!/usr/bin/env bash
# Watch the definitive pi05 + groot runs; on each NEW checkpoint, run the sampling
# reconstruction on BOTH the held-out val scenes and train scenes, and append a one-line
# summary to $RESULTS. Runs recon on GPU 6 (has headroom beside pi05's 18GB); Monitor the
# RESULTS file for the numbers. Loops until killed (TaskStop).
set -u
REPO=/home/nas5/jooyeolyun/repos/DronePhotographer
RESULTS=${RESULTS:-/tmp/recon_results.log}
SEEN=${SEEN:-/tmp/recon_seen.txt}
GPU=${GPU:-6}
N=${N:-48}
cd "$REPO"
touch "$SEEN"

py_pi05=/home/jooyeolyun/anaconda3/envs/vla/bin/python
py_groot=/home/jooyeolyun/anaconda3/envs/vla_groot/bin/python

recon() {  # policy checkpoint split -> "Xcm Ydeg wZ%"
  local pol=$1 ck=$2 split=$3 py log line w20
  py=$([ "$pol" = pi05 ] && echo "$py_pi05" || echo "$py_groot")
  log=/tmp/recon_${pol}_${4}_${split}.log
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    "$py" scripts/check_reconstruction_lerobot.py \
    --policy "$pol" --checkpoint "$ck" --scenes "$split" --n "$N" --seed 0 > "$log" 2>&1
  line=$(grep "reconstruct SAMPLED" "$log" | grep -oE "[0-9.]+ cm +[0-9.]+ deg" | tr -s ' ' | head -1)
  w20=$(grep -oE "within 20cm: [0-9]+%" "$log" | grep -oE "[0-9]+%" | head -1)
  [ -z "$line" ] && line="FAILED (see $log)"
  echo "${line} w${w20}"
}

while true; do
  for pol in groot pi05; do
    rundir=runs/${pol}_drone
    for ck in $(ls -d ${rundir}/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort); do
      step=$(echo "$ck" | grep -oE "checkpoints/[0-9]+" | grep -oE "[0-9]+$")
      key="${pol}:${step}"
      grep -qxF "$key" "$SEEN" && continue
      # skip half-written checkpoints (driver polls faster than lerobot finishes saving);
      # do NOT mark seen so it retries next poll once model.safetensors lands.
      [ -f "$ck/model.safetensors" ] || ls "$ck"/model-*.safetensors >/dev/null 2>&1 || continue
      echo "$key" >> "$SEEN"
      echo "[$(date +%H:%M)] $pol step $step: reconstructing (val + train)..." >> "$RESULTS"
      v=$(recon "$pol" "$ck" val "$step")
      t=$(recon "$pol" "$ck" train "$step")
      echo "RECON $pol step=$step  VAL[$v]  TRAIN[$t]" >> "$RESULTS"
    done
  done
  sleep 60
done
