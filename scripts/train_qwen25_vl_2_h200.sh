#!/usr/bin/env bash
set -euo pipefail

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

CUDA_VISIBLE_DEVICES=4,6 \
torchrun --nproc_per_node=2 --master_port "${MASTER_PORT}" scripts/train_qwen25_vl.py \
  --config configs/qwen25_vl_7b_2xh200.yaml
