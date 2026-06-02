#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

LOG_DIR="runs/cifar10/flow/jit/x0target_vloss_heun/${X0LOOP_RUN_TIMESTAMP}/logs"
mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/launcher.log" 2>&1

export MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

echo "[x0loop] root=${ROOT}"
echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"

RESUME_ARGS=()
if [ "$#" -gt 0 ] && [[ "${1}" != --* ]]; then
  RESUME_ARGS+=(--set "train.resume=${1}")
  shift
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml \
  --runtime-config "${RUNTIME_CONFIG:-x0loop/configs/runtime/fsdp_checkpoint_compile.yaml}" \
  "${RESUME_ARGS[@]}" \
  "$@"
