#!/bin/bash
# Submit the cross-policy comparison: every policy, the same 64 episodes, the same loop.
#
#   bash scripts/submit_policy_comparison.sh            # submit all six
#   bash scripts/submit_policy_comparison.sh pi05 dp    # a subset
#   QOS=extra bash scripts/submit_policy_comparison.sh  # somewhere other than own
#
# What is held fixed, and why it has to be:
#   --episodes-from configs/eval_episodes_v1.json   the manifest; d_start is asserted to 1e-6
#   --render-res 1024 768                           the training-native render. Every policy
#                                                   here was exported from a 1024x768 render
#                                                   squashed to its own input (256^2 Cosmos,
#                                                   224^2 VLAs, 480x720 DP); rendering that
#                                                   and letting each adapter squash reproduces
#                                                   each policy's training frames exactly.
#   --stop-on-shoot 1 --shoot-threshold 0.5         the (fixed) arrival rule, same for all
#   --samples 1                                     one draw per chunk. K>1 is extra inference
#                                                   compute and gets its own, labelled row.
#   --seed 0                                        per-episode seeds derive from it
#
# Results land in runs/closed_loop/cmp_<policy>.json; join them with
#   .venv-analysis/bin/python scripts/compare_policies.py --run v6=runs/closed_loop/cmp_v6_12k.json ...
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
QOS="${QOS:-own}"
# `share` is spare capacity anywhere in the cluster and needs its own account; every other
# QOS runs under the default one. Preemption is cheap here because closed_loop_eval resumes
# from <out>.partial.json and the sbatch wrapper resubmits itself.
ACCT=(); [ "$QOS" = share ] && ACCT=(-A share)
TIME="${TIME:-10:00:00}"        # 64 episodes at ~3-5 min each; the 1024x768 render is the slow part
COMMON=(--episodes-from configs/eval_episodes_v1.json --render-res 1024 768
        --stop-on-shoot 1 --shoot-threshold 0.5 --samples 1 --seed 0)

declare -A CK CFG
V6=runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v6/checkpoints
V5=runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v5/checkpoints
CK[v6_12k]="$V6/iter_000012000";   CFG[v6_12k]="--policy cosmos"     # best held-out of v6 (pinned)
CK[v6_27k]="$V6/iter_000027000";   CFG[v6_27k]="--policy cosmos"     # the one that was asked for
CK[v5_49k]="$V5/iter_000049000";   CFG[v5_49k]="--policy cosmos"     # v5's best survivor; same rule as the rest
CK[pi05]="baseline_ckpts/pi05_fair/checkpoints/300000/pretrained_model";  CFG[pi05]="--policy pi05"
CK[groot]="baseline_ckpts/groot_fair/checkpoints/210000/pretrained_model"; CFG[groot]="--policy groot"
CK[autophoto]="baseline_ckpts/autophoto/autophoto_22000_steps.zip"
CFG[autophoto]="--policy autophoto --policy-config baseline_ckpts/weights/autophoto/resnet-model42.pt"
CK[dp]="baseline_ckpts/20260818_064526_diffusion_policy_dinov2/ckpt_best.pt"
CFG[dp]="--policy dp --policy-config baseline_ckpts/20260818_064526_diffusion_policy_dinov2/config.yaml"

ORDER=(v6_12k v6_27k v5_49k pi05 groot dp autophoto)
[ $# -gt 0 ] && ORDER=("$@")
for name in "${ORDER[@]}"; do
    [ -n "${CK[$name]:-}" ] || { echo "unknown policy '$name' (choose from ${!CK[*]})" >&2; exit 2; }
    [ -e "${CK[$name]}" ] || { echo "missing checkpoint for $name: ${CK[$name]}" >&2; exit 2; }
    # shellcheck disable=SC2086
    jid=$(sbatch "${ACCT[@]}" --qos="$QOS" --time="$TIME" --job-name="cmp_$name" scripts/sbatch_closed_loop_eval.sh \
            ${CFG[$name]} --checkpoint "${CK[$name]}" "${COMMON[@]}" \
            --out "runs/closed_loop/cmp_$name.json" | grep -oE '[0-9]+$')
    printf "  %-8s -> job %s\n" "$name" "$jid"
done
