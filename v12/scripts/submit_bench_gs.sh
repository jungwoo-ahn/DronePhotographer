#!/bin/bash
# Paper Table 5: every policy of the same-scene table on the 3DGS benchmark (real-world
# reconstructions), same loop and flags as benchmark v2, then the same scorer.
#
#   bash scripts/submit_bench_gs.sh              # from the account whose `own` quota runs it
#
# At most 4 GPUs at a time: three rollouts and the caption job start together, the T2A
# rollout follows its captions, and each of the other four rollouts follows one of the first
# four. Detect and perceptual run after all eight, the table after both.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
M=configs/eval_episodes_gs_v1.json
B=runs/bench_gs
CK=$V12/runs/train/cosmos3_camera_policy/action_sft
R="--qos=own --time=02:00:00"          # ~25 min a policy at 90 episodes
COMMON="--episodes-from $M --render-res 1024 768 --stop-on-shoot 1 --shoot-threshold 0.5 --samples 1 --seed 0"
EVAL=scripts/sbatch_closed_loop_eval.sh

.venv-analysis/bin/python - "$M" "$B/goal_frames.json" <<'EOF'
import json, sys
eps = json.load(open(sys.argv[1]))["episodes"]
frames = list(dict.fromkeys(e["goal_image"] for e in eps))
json.dump({"_what": f"Goal frames of {sys.argv[1]}, for caption_goal_frames.py --frames-from.",
           "episodes": len(eps), "distinct": len(frames), "frames": frames},
          open(sys.argv[2], "w"), indent=1)
print(f"{len(frames)} goal frames -> {sys.argv[2]}")
EOF

cap=$(sbatch --parsable --qos=own --job-name=gscap scripts/sbatch_caption_goals.sh --frames-from $B/goal_frames.json \
      --style instruct --resize 512 --models qwen3.5-4b --out $B/goal_captions.json)
ours=$(sbatch --parsable $R --job-name=gs_ours $EVAL --policy cosmos \
       --checkpoint $CK/camera_policy_nano_v6/checkpoints/iter_000045000 $COMMON --out $B/rollout_ours_v6_45k.json)
novis=$(sbatch --parsable $R --job-name=gs_novis $EVAL --policy cosmos \
        --checkpoint $CK/camera_policy_nano_v6_novision/checkpoints/iter_000045000 $COMMON --out $B/rollout_novis_45k.json)
i2a=$(sbatch --parsable $R --job-name=gs_i2a $EVAL --policy cosmos --goal-channel panel \
      --checkpoint $CK/camera_policy_nano_v6_goalimg/checkpoints/iter_000045000 $COMMON --out $B/rollout_i2a_goalimg_45k.json)
t2a=$(sbatch --parsable $R --job-name=gs_t2a --dependency=afterok:$cap $EVAL --policy cosmos --goal-channel caption \
      --captions $B/goal_captions.json \
      --checkpoint $CK/camera_policy_nano_v6_caption/checkpoints/iter_000045000 $COMMON --out $B/rollout_t2a_caption_45k.json)
pi05=$(sbatch --parsable $R --job-name=gs_pi05 --dependency=afterany:$ours $EVAL --policy pi05 \
       --checkpoint baseline_ckpts/pi05_fair/checkpoints/300000/pretrained_model $COMMON --out $B/rollout_pi05.json)
groot=$(sbatch --parsable $R --job-name=gs_groot --dependency=afterany:$novis $EVAL --policy groot \
        --checkpoint baseline_ckpts/groot_fair/checkpoints/210000/pretrained_model $COMMON --out $B/rollout_groot.json)
dp=$(sbatch --parsable $R --job-name=gs_dp --dependency=afterany:$i2a $EVAL --policy dp \
     --checkpoint baseline_ckpts/20260818_064526_diffusion_policy_dinov2/ckpt_best.pt \
     --policy-config baseline_ckpts/20260818_064526_diffusion_policy_dinov2/config.yaml $COMMON --out $B/rollout_dp.json)
auto=$(sbatch --parsable $R --job-name=gs_auto --dependency=afterany:$t2a $EVAL --policy autophoto \
       --checkpoint baseline_ckpts/autophoto/autophoto_22000_steps.zip \
       --policy-config baseline_ckpts/weights/autophoto/resnet-model42.pt $COMMON --out $B/rollout_autophoto.json)

ROLL=$ours:$novis:$i2a:$t2a:$pi05:$groot:$dp:$auto
RUNS="ours=$B/rollout_ours_v6_45k.json novis=$B/rollout_novis_45k.json i2a=$B/rollout_i2a_goalimg_45k.json \
t2a=$B/rollout_t2a_caption_45k.json pi05=$B/rollout_pi05.json groot=$B/rollout_groot.json dp=$B/rollout_dp.json \
autophoto=$B/rollout_autophoto.json"
SC="--manifest $M --dir $B"
det=$(sbatch --parsable --qos=own --gres=gpu:1 --cpus-per-gpu=8 --mem=64G --dependency=afterany:$ROLL \
      scripts/sbatch_score_benchmark.sh detect $SC --runs $RUNS)
per=$(sbatch --parsable --qos=own --dependency=afterany:$ROLL scripts/sbatch_perceptual.sh --manifest $M \
      --metrics dinov3 --runs $B/rollout_ours_v6_45k.json $B/rollout_novis_45k.json $B/rollout_i2a_goalimg_45k.json \
      $B/rollout_t2a_caption_45k.json $B/rollout_pi05.json $B/rollout_groot.json $B/rollout_dp.json \
      $B/rollout_autophoto.json --out $B/perceptual.json)
tab=$(sbatch --parsable --qos=own --cpus-per-task=4 --mem=32G --dependency=afterok:$det:$per \
      scripts/sbatch_score_benchmark.sh table $SC --runs $RUNS --ref ours --perceptual $B/perceptual.json)
echo "cap=$cap ours=$ours novis=$novis i2a=$i2a t2a=$t2a pi05=$pi05 groot=$groot dp=$dp auto=$auto"
echo "detect=$det perceptual=$per table=$tab"
