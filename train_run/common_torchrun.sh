#!/usr/bin/env bash

# Shared torchrun helpers for local single-node experiments.
# Usage:
#   source train_run/common_torchrun.sh
#   setup_torchrun_env
#   torchrun ... --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" ...

find_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

setup_torchrun_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}" 
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-$(find_free_port)}"
  echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}
