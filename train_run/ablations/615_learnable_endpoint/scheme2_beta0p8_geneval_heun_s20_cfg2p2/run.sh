#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${ROOT}"

RUN_NAME="scheme2_beta0p8_geneval_heun_s20_cfg2p2"
CONFIG_PATH="${ROOT}/train_run/ablations/615_learnable_endpoint/${RUN_NAME}/config.yaml"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${ROOT}/x0loop/configs/runtime/ddp_checkpoint_compile.yaml}"
TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_DIR="${ROOT}/runs/ablations/cifar10_flow_x0_vloss/jit/learnable_endpoint/${RUN_NAME}/${TIMESTAMP}"
LOG_DIR="${OUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/launcher.log"
PID_FILE="${LOG_DIR}/launcher.pid"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((47000 + RANDOM % 10000))}"
export X0LOOP_RUN_TIMESTAMP="${TIMESTAMP}"

if [ -z "${NPROC_PER_NODE:-}" ]; then
  NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi
export NPROC_PER_NODE

{
  echo "[x0loop] root=${ROOT}"
  echo "[ablation] name=${RUN_NAME}"
  echo "[ablation] config=${CONFIG_PATH}"
  echo "[ablation] out_dir=${OUT_DIR}"
  echo "[torchrun] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_PORT=${MASTER_PORT}"
} >> "${LOG_FILE}" 2>&1

CMD=(train_run/run_and_plot.sh "${LOG_FILE}" "${TIMESTAMP}" "${ROOT}" -- \
  uv run torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config "${CONFIG_PATH}" \
  --runtime-config "${RUNTIME_CONFIG}" \
  --set "logging.out_dir=${OUT_DIR}" \
  "$@")

if [ "${X0LOOP_RUN_BACKGROUND:-1}" = "1" ]; then
  setsid "${CMD[@]}" >/dev/null 2>&1 &
  RUN_PID=$!
  echo "${RUN_PID}" > "${PID_FILE}"
  {
    echo "[x0loop] started in background"
    echo "[x0loop] pid=${RUN_PID}"
    echo "[x0loop] log=${LOG_FILE}"
  } >> "${LOG_FILE}" 2>&1
  cat <<EOF
==================================================================
[x0loop] ${RUN_NAME} started in background (GPU=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE}, pid=${RUN_PID})
  log        : ${LOG_FILE}
  tail       : tail -f ${LOG_FILE}
  process    : ps -fp ${RUN_PID}
  stop group : kill -- -${RUN_PID}
==================================================================
EOF
else
  "${CMD[@]}" >/dev/null 2>&1
fi
