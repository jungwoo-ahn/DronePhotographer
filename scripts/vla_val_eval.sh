#!/usr/bin/env bash
# Held-out-val eval for a LeRobot VLA: (1) sampling/reconstruction and (2) closed-loop sim
# (own-goal) on the 8 val scenes. Usage: vla_val_eval.sh <pi05|groot> <ckpt_dir> <gpu> <env>
set -u
POL=$1; CK=$2; GPU=$3; ENVN=${4:-vla_groot}
cd /home/nas5/jooyeolyun/repos/DronePhotographer
PY=/home/jooyeolyun/anaconda3/envs/$ENVN/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
V6=data/vlm_object_placing_v6_260428_061326
RES=/tmp/${POL}_val_eval.txt; : > "$RES"

echo "=== $POL VAL reconstruction (sampling) | ckpt $CK ===" >> "$RES"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/check_reconstruction_lerobot.py \
  --policy "$POL" --checkpoint "$CK" --scenes val --n 96 --seed 0 --shuffle_goals 1 2>/dev/null \
  | grep -E "reconstruct SAMPLED|within 20cm|action MSE|GOAL-DEPEND|recon cm|Δaction" >> "$RES"

echo "=== $POL VAL closed-loop sim (own-goal, 1 mapped placement / val scene) ===" >> "$RES"
mapped(){ $PY -c "import json,sys;from pathlib import Path;fm=json.load(open('configs/policy/facing_map_final.json'));d=json.load(open('$1'));sys.exit(0 if (fm.get(Path(d['object_file']).stem) or {}).get('front_az') is not None else 1)"; }
for sc in Chill-Camping_150d7263 Modern-Warehouse_a5eea495 Modern-Office-Full-set-01_d2deb1e1 \
          Parking_70ae1f27 Nature-Snowy-Mountain-Village-Retreat_f884ef3e \
          basement_8f9ffd5b-654b-4efe-9f09-7df7e49d2ab8 Desert-Roadside-Repair-Garage_a1a2b2ac \
          office-buildingwarehouse-creative-headquarters; do
  for d in data/trajectories_full/${sc}__*; do
    name=$(basename "$d"); [ -f "$d/data.json" ] && [ -f "$V6/$name.json" ] || continue
    mapped "$d/data.json" || continue
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rollout_vla.py --policy "$POL" --checkpoint "$CK" \
      --data_json "$d/data.json" --v6_json "$V6/$name.json" --own_goal \
      --out_dir /tmp/${POL}_valsim/$name --blender blender/blender --max_steps 24 \
      > /tmp/${POL}_valsim_$name.log 2>&1
    echo "$name | $(grep '^END:' /tmp/${POL}_valsim_$name.log | tail -1)" >> "$RES"
    break
  done
done
echo "=== $POL VAL EVAL DONE ===" >> "$RES"
