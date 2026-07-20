#!/bin/bash
# Cosmos v8 — extra QOS + self-requeue on preempt (runs now on spare capacity).
#
# extra is preemptible by other users' `own` jobs (CANCEL mode, ~5 min grace).
# On preempt we resubmit ourselves with --resume <latest run dir>, so training
# continues from the last checkpoint (save_iter=1000). qos=extra is baked in
# (NOT a CLI override) so the self-resubmit also lands on extra.
#
#   sbatch scripts/sbatch_cosmos_2b_v8_extra.sh
#
#SBATCH --job-name=cosmos_v8
#SBATCH --gres=gpu:2
#SBATCH --qos=extra             # preemptible; baked in so the requeue stays on extra
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=360G
#SBATCH --time=05:00:00
#SBATCH --output=runs/slurm-%x-%j.out

set -uo pipefail                # NOT -e: keep the trap alive across the signalled wait

PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source .venv/bin/activate
export PYTHONPATH="$PROJ"
export HF_HOME=/home/nas_main/.cache/huggingface

_handle_preempt() {
    # Resubmit ONLY on a real preemption. CANCEL-mode preempt shows PREEMPTED or
    # "CANCELLED by 0" (system); a user `scancel` shows "CANCELLED by <uid>" and
    # must NOT trigger a resubmit (else the job is un-killable).
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    echo "[preempt-handler] job=$SLURM_JOB_ID state='$state'"
    if [[ "$state" == "PREEMPTED" || "$state" == "CANCELLED by 0" ]]; then
        # newest v8 run dir that actually has a checkpoint to resume from
        latest=$(ls -dt runs/*_cosmos_2b_v8 2>/dev/null | while read -r d; do
                     [[ -f "$d/ckpt_last.pt" ]] && { echo "$d"; break; }; done)
        if [[ -n "$latest" ]]; then
            echo "[preempt-handler] resubmitting --resume $latest"
            sbatch "$0" --resume "$latest"
        else
            echo "[preempt-handler] no checkpoint yet; resubmitting fresh"
            sbatch "$0"
        fi
    fi
    exit 143
}
trap _handle_preempt SIGTERM

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2

# Background + wait so SIGTERM reaches the trap immediately (a foreground child
# would defer the handler until it exits).
torchrun --standalone --nproc_per_node=2 \
    scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b_v8.yaml "$@" &
wait $!
