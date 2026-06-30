#!/bin/bash
# Closed-loop Blender eval of a trained Cosmos policy on held-out val scenes.
#
#   sbatch scripts/sbatch_rollout_eval.sh --checkpoint runs/<ts>_cosmos_2b/ckpt_last.pt \
#       --num-placements 4 --max-steps 16
#
# Needs GPU (policy) + Blender (renders). During training (job using the 2 own GPUs)
# this must be `extra` (1 extra GPU); after training finishes, `--qos=own` also works.
# Blender runs headless from `blender/blender`; its missing system libs (libGL/
# libXfixes/libxkbcommon) are supplied from blender/syslibs (NAS) via LD_LIBRARY_PATH
# below — works on login and bare workers. (If actual Cycles rendering still fails on a
# worker, fall back to `--container=` with an image that ships Mesa/GL.)
#
#SBATCH --job-name=cosmos_rollout_eval
#SBATCH --gres=gpu:1
#SBATCH --qos=extra
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%x-%j.out

set -euo pipefail
PROJ="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ"
source .venv/bin/activate
export PYTHONPATH="$PROJ"
export HF_HOME=/home/nas_main/.cache/huggingface
# Blender's missing system libs (login + bare worker image lack them) — from NAS.
export LD_LIBRARY_PATH="$PROJ/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

python scripts/rollout_eval.py "$@"
