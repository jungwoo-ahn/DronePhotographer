#!/bin/bash
# One more checkpoint through both paper benchmarks -- benchmark v2 (Tables 1-3) and the 3DGS set
# (Table 5) -- with exactly the flags of the runs already in those tables, then re-score each
# table with the new row added. Existing rollouts are not re-run; their detections are cached.
#
#   bash scripts/submit_ckpt_eval.sh ours12k \
#       runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v6/checkpoints/iter_000012000
#   (extra flags after the checkpoint go to closed_loop_eval, e.g. --goal-channel panel)
set -euo pipefail
LABEL=$1; CKPT=$2; shift 2
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
[ -d "$CKPT" ] || { echo "no checkpoint $CKPT" >&2; exit 1; }
EVAL=scripts/sbatch_closed_loop_eval.sh
FLAGS="--render-res 1024 768 --stop-on-shoot 1 --shoot-threshold 0.5 --samples 1 --seed 0"

# label=run-file pairs already in each table; the new run is appended to both.
V2_RUNS="ours=rollout_ours_v6_45k novis=rollout_novis_45k i2a=rollout_i2a_goalimg_45k t2a=rollout_t2a_caption_45k \
pi05=rollout_pi05 groot=rollout_groot dp=rollout_dp autophoto=rollout_autophoto"
GS_RUNS="$V2_RUNS"

for bench in v2 gs; do
    if [ $bench = v2 ]; then
        M=configs/eval_episodes_v2.json; B=runs/bench_v2; T=04:00:00; EXTRA="--render-profile v7"; RUNS=$V2_RUNS
    else
        M=configs/eval_episodes_gs_v1.json; B=runs/bench_gs; T=01:30:00; EXTRA=""; RUNS=$GS_RUNS
    fi
    OUT=$B/rollout_$LABEL.json
    # Checkpoints added by earlier runs of this script, so a re-score keeps every row it has had.
    REG=$B/extra_runs.txt
    EXTRA_RUNS=$(grep -v "^$LABEL=" "$REG" 2>/dev/null || true)
    roll=$(sbatch --parsable --qos=own --time=$T --job-name=${bench}_$LABEL $EVAL --policy cosmos \
           --checkpoint "$CKPT" --episodes-from $M $FLAGS $EXTRA "$@" --out $OUT)
    LR=""; FILES=""
    for p in $RUNS $EXTRA_RUNS; do LR="$LR ${p%%=*}=$B/${p#*=}.json"; FILES="$FILES $B/${p#*=}.json"; done
    LR="$LR $LABEL=$OUT"; FILES="$FILES $OUT"          # the new row last
    grep -q "^$LABEL=" "$REG" 2>/dev/null || echo "$LABEL=rollout_$LABEL" >> "$REG"
    det=$(sbatch --parsable --qos=own --gres=gpu:1 --cpus-per-gpu=8 --mem=64G --dependency=afterok:$roll \
          scripts/sbatch_score_benchmark.sh detect --manifest $M --dir $B --runs $LR)
    per=$(sbatch --parsable --qos=own --dependency=afterok:$roll scripts/sbatch_perceptual.sh --manifest $M \
          --metrics dinov3 --runs $FILES --out $B/perceptual.json)
    tab=$(sbatch --parsable --qos=own --cpus-per-task=4 --mem=32G --dependency=afterok:$det:$per \
          scripts/sbatch_score_benchmark.sh table --manifest $M --dir $B --runs $LR --ref ours \
          --perceptual $B/perceptual.json)
    echo "$bench: rollout=$roll detect=$det perceptual=$per table=$tab -> $B/table.md"
done
