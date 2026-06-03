#!/usr/bin/env bash
# Usage:
#   bash train_run/train_533/train_x0loop_cifar10_unet_weighted_x0.sh [ckpt.pt] [extra x0loop.train args...]
# Logs:
#   runs/launcher_logs/train_533_unet_weighted_x0/<timestamp>/logs/launcher.log
# Starts in background and prints the log path plus the command to stop it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

LOG_DIR="runs/launcher_logs/train_533_unet_weighted_x0/${X0LOOP_RUN_TIMESTAMP}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launcher.log"

export MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

{
  echo "[x0loop] root=${ROOT}"
  echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
} >> "${LOG_FILE}" 2>&1

RESUME_ARGS=()
if [ "$#" -gt 0 ] && [[ "${1}" != --* ]]; then
  RESUME_ARGS+=(--set "train.resume=${1}")
  shift
fi

setsid train_run/run_and_plot.sh "${LOG_FILE}" "${X0LOOP_RUN_TIMESTAMP}" "${ROOT}" -- \
  torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config train_run/configs/cifar10/cifar10_unet_flow_train_x0_weighted.yaml \
  --runtime-config "${RUNTIME_CONFIG:-x0loop/configs/runtime/fsdp_checkpoint_compile.yaml}" \
  "${RESUME_ARGS[@]}" \
  "$@" >> "${LOG_FILE}" 2>&1 &
RUN_PID=$!

echo "[x0loop] started in background"
echo "[x0loop] log: ${ROOT}/${LOG_FILE}"
echo "[x0loop] pid: ${RUN_PID}"
echo "[x0loop] stop: kill -- -${RUN_PID}"
echo "[x0loop] watch: tail -f ${ROOT}/${LOG_FILE}"
echo "[x0loop] plots: generated automatically after successful completion; see log for paths"
