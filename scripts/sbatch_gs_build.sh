#!/bin/bash
#SBATCH --job-name=gsbuild
#SBATCH --time=03:00:00
#SBATCH --output=runs/gsbuild_%j.out
# Blender and gsplat both need the driver-carrying image the rollout renders use.
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# One stage of scripts/gs_build_episodes.py (3DGS benchmark episodes). Resources from the submit:
#   sbatch --qos=own --cpus-per-task=16 --mem=96G scripts/sbatch_gs_build.sh geom
#   sbatch --qos=own --gres=gpu:1 --cpus-per-gpu=8 --mem=96G scripts/sbatch_gs_build.sh candidates
set -euo pipefail
umask 002          # submitted from jooyeolyun; keep outputs editable by jungwooahn
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
cd "$V12"
NV=$(find "$V12"/.venv-gs/lib/python3*/site-packages/nvidia -maxdepth 2 -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:$NV${LD_LIBRARY_PATH:-}"
# The pod's CPU quota is not a cpuset: unpinned pools size to all 144 host cores and throttle.
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8} \
       OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
exec .venv-gs/bin/python scripts/gs_build_episodes.py "$@"
