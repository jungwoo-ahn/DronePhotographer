#!/bin/bash
#SBATCH --job-name=benchscore
#SBATCH --time=02:00:00
#SBATCH --output=runs/benchscore_%j.out
# Run one stage of scripts/score_benchmark.py. Resources come from the submit line:
#   sbatch --qos=own --cpus-per-task=16 --mem=96G scripts/sbatch_score_benchmark.sh geom
#   sbatch --qos=own --gres=gpu:1 --cpus-per-gpu=8 --mem=64G scripts/sbatch_score_benchmark.sh detect --runs ...
set -euo pipefail
umask 002          # submitted from jooyeolyun; keep outputs editable by jungwooahn
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
exec .venv-analysis/bin/python scripts/score_benchmark.py "$@"
