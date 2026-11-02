#!/bin/bash
#SBATCH --job-name=goalcap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/home/nas_main/jungwooahn/projects/DronePhotographerV12/runs/goalcap_%j.out
# VLM captions for goal frames -- the text channel of the caption-conditioned arm.
#
#   sbatch --qos=own scripts/sbatch_caption_goals.sh --n 60 --models qwen3.5-4b qwen3vl-8b
#
# The default ubuntu:22.04 worker has no compiler and transformers JITs nothing here, so no
# --container is needed; the venv carries torch and its CUDA runtime. ~30 s a caption on the
# Qwen3.5 linear-attention fallback, so 60 frames x 2 models is about an hour.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
export HF_HOME=/home/nas_main/.cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
exec .venv-vla/bin/python scripts/caption_goal_frames.py "$@"
