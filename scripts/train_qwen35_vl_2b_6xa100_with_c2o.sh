#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-1,2,3,4,5,6}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ID_ARRAY[@]}}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(
    python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
  )"
fi

export MASTER_PORT

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" scripts/train.py \
  --config configs/qwen35_vl_2b_6xa100_with_c2o.yaml
