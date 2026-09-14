#!/bin/bash
#SBATCH --job-name=cleval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
# 3h was sized for 12-24 episodes. A 160-episode sweep runs ~8h at ~3 min each, and
# silently TIMED OUT at 63/160. Resume makes a restart cheap, but the limit should
# not be the thing that decides how many episodes finish.
#SBATCH --time=12:00:00
#SBATCH --output=runs/cleval_%j.out
# Needs a C compiler for torch inductor (worker pods are driver-only), same as training.
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# Closed-loop evaluation of a trained camera policy.
#
#   sbatch --qos=extra scripts/sbatch_closed_loop_eval.sh \
#       --checkpoint runs/train/.../checkpoints/iter_000005000 \
#       --episodes 24 --out runs/closed_loop/iter5000.json
#
#   # a baseline, same loop, same episodes -- only the adapter differs:
#   sbatch --qos=own scripts/sbatch_closed_loop_eval.sh --policy pi05 \
#       --checkpoint baseline_ckpts/pi05_fair/checkpoints/300000/pretrained_model \
#       --episodes-from configs/eval_episodes_v1.json --render-res 1024 768 \
#       --out runs/closed_loop/cmp_pi05.json
#
# Inference is single-GPU regardless of the training sharding: DCP reshards on load,
# so a 2-GPU-sharded checkpoint loads onto one device with no conversion step.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
CF="$V12/repos/cosmos-framework"
cd "$V12"

# ONE script, TWO venvs. `--policy cosmos` (the default) needs the cosmos-framework venv;
# the baselines (`pi05`, `groot`, `dp`) need lerobot + transformers 5, which cannot coexist
# with the framework's pins, so they live in .venv-vla. The eval script itself imports
# neither stack at module level and only pulls in the one its adapter needs.
_POLICY=cosmos
_prev=""
for _a in "$@"; do
    case "$_prev" in --policy) _POLICY="$_a" ;; esac
    case "$_a" in --policy=*) _POLICY="${_a#*=}" ;; esac
    _prev="$_a"
done
if [ "$_POLICY" = cosmos ]; then
    PY="$CF/.venv/bin/python"
    SP="$CF/.venv/lib/python3.13/site-packages"
    bash "$V12/scripts/setup_cosmos_env.sh"      # cuda_cudart symlink + experiment registration
    export PYTHONPATH="$V12:$CF:${PYTHONPATH:-}"
else
    PY="$V12/.venv-vla/bin/python"
    SP="$V12/.venv-vla/lib/python3.12/site-packages"
    export PYTHONPATH="$V12:${PYTHONPATH:-}"
    # Everything the baselines pull from the Hub is in the shared cache; a worker has no
    # business going online, and a gated backbone would 403 anyway.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
echo "policy=$_POLICY  python=$PY"

# The CUDA libs ship inside the venv's nvidia/* wheels but are not on the loader path.
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# Blender's own missing system libs, for the rollout renderer.
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:$LD_LIBRARY_PATH"
export HF_HOME=/home/nas_main/.cache/huggingface
export WAN_VAE_PATH="$CF/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
# The recipe interpolates ${oc.env:CAMERA_ROOT} when the config is built. Inference
# never reads the dataset, but the interpolation still has to resolve or the config
# raises before the model is even constructed.
# Must be the v5 export: its action width is 10, and lerobot_v1's is 9, so the recipe would
# describe a different model than the checkpoint. Inference never reads the dataset, but the
# ${oc.env:CAMERA_ROOT} interpolation has to resolve before the model is constructed. Same
# reasoning as scripts/sbatch_shoot_probe.sh, which was already on v5 while this was not.
export CAMERA_ROOT="${CAMERA_ROOT:-$V12/runs/lerobot_v5}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$V12/runs/train}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$CF/examples/checkpoints/Cosmos3-Nano}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Self-requeue on preemption. This eval takes ~4.5 h and `share` has been cutting it at
# ~30 min, so without this it never finishes. Safe because closed_loop_eval resumes from
# <out>.partial.json — each attempt continues rather than restarting.
#
# Three pitfalls the cluster guide warns about, all handled:
#   - `exec` would replace this shell and destroy the trap  -> run python in background
#   - a FOREGROUND child defers the handler until it exits  -> `&` + `wait`
#   - `scancel` raises the same SIGTERM as a preemption      -> check sacct State first
_args=("$@")
_handle_term() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        PREEMPTED*|*CANCELLED*by\ 0*)
            echo "[requeue] preempted (state=$state) — resubmitting to finish the rest" >&2
            sbatch -A share --qos=share "$0" "${_args[@]}" >&2
            ;;
        *) echo "[requeue] state=$state — not a preemption, exiting" >&2 ;;
    esac
    exit 143
}
trap _handle_term SIGTERM

"$PY" scripts/closed_loop_eval.py "$@" &
wait $!
