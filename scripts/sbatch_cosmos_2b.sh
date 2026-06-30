#!/bin/bash
# Cosmos goal-conditioned policy — 2x B200 DDP training (free-tier own QOS).
#
# Full run:            sbatch scripts/sbatch_cosmos_2b.sh
# Short validation:    sbatch --time=04:00:00 --job-name=cosmos_val \
#                          scripts/sbatch_cosmos_2b.sh --max_iter 2000 --warmup_iter 200
# Resume (after stop): sbatch scripts/sbatch_cosmos_2b.sh --resume runs/<ts>_cosmos_2b
#
# Extra args after the script name are forwarded to train_cosmos_policy.py
# (--max_iter / --warmup_iter / --max_samples / --resume).
#
#SBATCH --job-name=cosmos_policy_2b
#SBATCH --gres=gpu:2
#SBATCH --qos=own               # free-own: 2 GPU, not preemptible
#SBATCH --cpus-per-gpu=12       # 24 CPU total (own cap 28; NUMA-local), feeds 2x8 dataloader workers
#SBATCH --mem=360G              # own cap ~439G; generous headroom, no pod OOM
#SBATCH --time=2-12:00:00       # placeholder — tighten from the smoke it/s before the full run
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
    scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml "$@"
