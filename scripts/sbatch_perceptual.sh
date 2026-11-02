#!/bin/bash
#SBATCH --job-name=perceptual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/perceptual_%j.out
# Final frame vs goal frame similarity (scripts/perceptual_eval.py). Weights come from the
# shared HF cache; a worker has no business going online.
#
#   sbatch --qos=own scripts/sbatch_perceptual.sh --manifest configs/eval_episodes_v2.json \
#       --metrics dinov3 --runs runs/bench_v2/rollout_*.json --out runs/bench_v2/perceptual.json
set -euo pipefail
umask 002
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
export HF_HOME=/home/nas_main/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SP="$V12/.venv-vla/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
exec .venv-vla/bin/python scripts/perceptual_eval.py "$@"
