#!/bin/bash
# Serve the cosmos3 camera policy from a checkout of this repo.
#
#   CUDA_VISIBLE_DEVICES=2 bash scripts/serve_cosmos_policy.sh [serve_policy.py args]
#
# Expects `repos/cosmos-framework` at the commit the checkpoint trained on, installed with
#   uv sync --frozen --all-extras --group=cu128-train
# The checkpoint and its VAE stay where training left them; nothing here writes to them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CF="$ROOT/repos/cosmos-framework"
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SP="$CF/.venv/lib/python3.13/site-packages"

: "${POLICY_CKPT:=$V12/runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v6/checkpoints/iter_000045000}"
: "${WAN_VAE_PATH:=$V12/repos/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${BASE_CHECKPOINT_PATH:=$V12/repos/cosmos-framework/examples/checkpoints/Cosmos3-Nano}"
: "${HF_HOME:=/home/nas_main/.cache/huggingface}"
: "${POLICY_OUTPUT:=$ROOT/runs/serve_policy}"

# transformer_engine looks for cuda_cudart, but torch cu128 installs it as cuda_runtime.
[ -e "$SP/nvidia/cuda_cudart" ] || ln -s cuda_runtime "$SP/nvidia/cuda_cudart"

export WAN_VAE_PATH BASE_CHECKPOINT_PATH HF_HOME
export IMAGINAIRE_OUTPUT_ROOT="$POLICY_OUTPUT"
export PYTHONPATH="$ROOT:$CF:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$CF"
exec "$CF/.venv/bin/python" "$ROOT/scripts/serve_policy.py" \
    --checkpoint "$POLICY_CKPT" --vae "$WAN_VAE_PATH" --output-dir "$POLICY_OUTPUT" "$@"
