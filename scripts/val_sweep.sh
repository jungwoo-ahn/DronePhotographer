#!/usr/bin/env bash
# Validation sweep: run the held-out-scene recon metric over EVERY saved checkpoint of a run,
# print the curve, and write it to TensorBoard as val/* scalars.
#
# Why: lerobot-train computes no held-out loss (env_eval_freq=0, and --dataset.eval_split would
# slice the TRAIN set, not our scene-disjoint split). Without this the VLAs would be reported at
# their FINAL checkpoint while DP is reported at a val-selected ckpt_best — an unfair asymmetry.
# This sweep gives the VLAs the same val-selected treatment and shows overfitting onset.
#
#   val_sweep.sh <run_name> <pi05|groot> <conda_env> "<gpu list>"
#   val_sweep.sh groot_fair groot vla_groot "1 2"
set -u
RUN=$1; POL=$2; ENVN=$3; read -r -a GPUS <<< "${4:-6 7}"
cd /home/nas5/jooyeolyun/repos/DronePhotographer
PY=/home/jooyeolyun/anaconda3/envs/$ENVN/bin/python
OUT=/tmp/${RUN}_sweep; mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.

i=0
for ck in runs/"$RUN"/checkpoints/[0-9]*; do
  [ -d "$ck/pretrained_model" ] || continue
  step=$(basename "$ck")
  [ -s "$OUT/$step.txt" ] && continue          # resumable: skip already-swept checkpoints
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/check_reconstruction_lerobot.py \
      --policy "$POL" --checkpoint "$ck/pretrained_model" --scenes val --n 96 --seed 0 \
      2>/dev/null | grep -E "reconstruct SAMPLED|within 20cm" > "$OUT/$step.txt" &
  i=$((i+1))
  [ $((i % ${#GPUS[@]})) -eq 0 ] && wait      # keep at most one job per GPU in flight
done
wait

# ingest with the `drone` env: it is the only one with tensorboard installed
/home/jooyeolyun/anaconda3/envs/drone/bin/python scripts/tb_val_ingest.py "$RUN" "$OUT"
echo "=== ${RUN} SWEEP DONE ==="
