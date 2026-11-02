#!/bin/bash
#SBATCH --job-name=evalv2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/evalv2_%j.out
# Build configs/eval_episodes_v2.json (scripts/build_eval_episodes_v2.py). One GPU for the person
# detector over ~10k goal and start frames; it falls back to the CPU, slowly, if the GPU is absent.
#
#   sbatch --qos=own scripts/sbatch_build_eval_v2.sh
set -euo pipefail
umask 002          # submitted from jooyeolyun; keep outputs editable by jungwooahn
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
exec .venv-analysis/bin/python scripts/build_eval_episodes_v2.py "$@"
