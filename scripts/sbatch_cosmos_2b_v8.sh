#!/bin/bash
# Cosmos goal-conditioned policy — v8 bundled run (2x B200 DDP, free-own).
#
# Bundles four levers to break the goal->action saturation (~21% of the DP ceiling):
#   clamp-fix (recovered dolly-in goals) + n_goal_tokens 32 + CFG-dropout + object-disjoint val.
# K=32 trains fresh (not resumable from any K=4 checkpoint). CFG is evaluated at
# inference on the resulting checkpoint (null / flip negatives, guidance sweep).
#
#   sbatch scripts/sbatch_cosmos_2b_v8.sh
#   sbatch scripts/sbatch_cosmos_2b_v8.sh --resume runs/<ts>_cosmos_2b_v8   # extend
#
#SBATCH --job-name=cosmos_v8
#SBATCH --gres=gpu:2
#SBATCH --qos=own               # free-own: 2 GPU, not preemptible
#SBATCH --cpus-per-gpu=12       # 24 CPU total (own cap 28; NUMA-local), feeds 2x8 dataloader workers
#SBATCH --mem=360G              # own cap ~439G; generous headroom, no pod OOM
#SBATCH --time=05:00:00         # ~4000 iter @ ~0.35 it/s (2-GPU DDP) ~190 min + val/viz/load headroom
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
    scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b_v8.yaml "$@"
