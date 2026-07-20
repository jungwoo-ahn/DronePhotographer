#!/bin/bash
# Cosmos goal-conditioned policy — RMSNorm goal-token fix probe (2x B200 DDP, free-own).
#
# Tests the conditioner fix: goal tokens are RMSNorm'd to the anchor's per-token scale
# (goal/anchor magnitude 0.0006 -> 1.0) so the goal is finally audible in cross-attention.
# Short 2000-iter run, saves every 500, to watch check_goal_dependence ratio move off 0.02.
#
#   sbatch scripts/sbatch_cosmos_2b_goalnorm.sh
#   sbatch scripts/sbatch_cosmos_2b_goalnorm.sh --resume runs/<ts>_cosmos_2b_goalnorm   # extend
#
#SBATCH --job-name=cosmos_goalnorm
#SBATCH --gres=gpu:2
#SBATCH --qos=own               # free-own: 2 GPU, not preemptible
#SBATCH --cpus-per-gpu=12       # 24 CPU total (own cap 28; NUMA-local), feeds 2x8 dataloader workers
#SBATCH --mem=360G              # own cap ~439G; generous headroom, no pod OOM
#SBATCH --time=02:30:00         # ~2000 iter @ ~0.35 it/s (2-GPU DDP) ~95 min + val/load headroom
#SBATCH --output=runs/slurm-%x-%j.out

set -euo pipefail

PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source .venv/bin/activate
export PYTHONPATH="$PROJ"
# Shared cache holds the gated HF token + Cosmos weights so workers pull warm.
export HF_HOME=/home/nas_main/.cache/huggingface

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2

# Single node, 2 GPUs: torchrun spawns one process per GPU (LOCAL_RANK -> cuda:N).
# No srun needed; --standalone runs its own localhost rendezvous inside the pod.
torchrun --standalone --nproc_per_node=2 \
    scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b_goalnorm.yaml "$@"
