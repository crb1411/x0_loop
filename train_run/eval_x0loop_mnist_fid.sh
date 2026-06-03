#!/usr/bin/env bash
# Usage:
#   bash train_run/eval_x0loop_mnist_fid.sh [extra x0loop.eval_mnist_fid args...]
# Logs:
#   runs/launcher_logs/eval_x0loop_mnist_fid/<timestamp>/logs/launcher.log
# Starts in background and prints the log path plus the command to stop it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

LOG_DIR="runs/launcher_logs/eval_x0loop_mnist_fid/${X0LOOP_RUN_TIMESTAMP}/logs"
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

setsid torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.eval_mnist_fid \
  --config x0loop/configs/mnist_dit_diffusion.yaml \
  --runtime-config "${RUNTIME_CONFIG:-x0loop/configs/runtime/fsdp_checkpoint_compile.yaml}" \
  --ckpt /data/seek/aigc/x0_loop/runs/mnist_diffusion/run_20260213_232141/checkpoints/ckpt_step_03280000.pt \
  --out-dir-base /data/seek/aigc/x0_loop/runs/mnist_diffusion/fid_mnist \
  --num-samples 10000 \
  --steps 100 \
  --sampler auto \
  "$@" >> "${LOG_FILE}" 2>&1 &
RUN_PID=$!

echo "[x0loop] started in background"
echo "[x0loop] log: ${ROOT}/${LOG_FILE}"
echo "[x0loop] pid: ${RUN_PID}"
echo "[x0loop] stop: kill -- -${RUN_PID}"
echo "[x0loop] watch: tail -f ${ROOT}/${LOG_FILE}"
