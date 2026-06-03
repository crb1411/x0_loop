#!/usr/bin/env bash
# Shared launcher for CIFAR10 flow x0-target v-loss ablations.
# Usage:
#   MODEL_NAME=dit ABLATION_TAG=unweighted CONFIG_PATH=... EXTRA_SETS=(...) bash .../_run_ablation.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${ABLATION_TAG:?ABLATION_TAG is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
if ! declare -p EXTRA_SETS >/dev/null 2>&1; then
  EXTRA_SETS=()
fi

find_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

setup_repo() {
  ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  cd "${ROOT}"
  export ROOT
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
}

setup_runtime_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-$(find_free_port)}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
}

setup_output_paths() {
  OUT_DIR="runs/ablations/cifar10_flow_x0_vloss/${MODEL_NAME}/${ABLATION_TAG}/${X0LOOP_RUN_TIMESTAMP}"
  LOG_DIR="${OUT_DIR}/logs"
  LOG_FILE="${LOG_DIR}/launcher.log"
  mkdir -p "${LOG_DIR}"
  export OUT_DIR LOG_DIR LOG_FILE
}

RESUME_ARGS=()
COMMON_ARGS=(
  --set "train.epochs=300"
  --set "process.sampler=heun"
  --set "sample.sampler=heun"
)

parse_resume_arg() {
  if [ "$#" -gt 0 ] && [[ "${1}" != --* ]]; then
    RESUME_ARGS+=(--set "train.resume=${1}")
    shift
  fi
  REMAINING_ARGS=("$@")
}

write_launch_header() {
  {
    echo "[x0loop] root=${ROOT}"
    echo "[ablation] model=${MODEL_NAME} tag=${ABLATION_TAG}"
    echo "[ablation] config=${CONFIG_PATH}"
    echo "[ablation] out_dir=${OUT_DIR}"
    echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
  } >> "${LOG_FILE}" 2>&1
}

start_training() {
  setsid train_run/run_and_plot.sh "${LOG_FILE}" "${X0LOOP_RUN_TIMESTAMP}" "${ROOT}" -- \
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m x0loop.train \
    --config "${CONFIG_PATH}" \
    --runtime-config "${RUNTIME_CONFIG:-x0loop/configs/runtime/fsdp_checkpoint_compile.yaml}" \
    "${COMMON_ARGS[@]}" \
    --set "logging.out_dir=${OUT_DIR}" \
    "${EXTRA_SETS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${REMAINING_ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
  RUN_PID=$!
}

print_status() {
  echo "[x0loop] started in background"
  echo "[x0loop] ablation: ${MODEL_NAME}/${ABLATION_TAG}"
  echo "[x0loop] log: ${ROOT}/${LOG_FILE}"
  echo "[x0loop] pid: ${RUN_PID}"
  echo "[x0loop] stop: kill -- -${RUN_PID}"
  echo "[x0loop] watch: tail -f ${ROOT}/${LOG_FILE}"
  echo "[x0loop] plots: generated automatically after successful completion; see log for paths"
}

setup_repo
setup_runtime_env
setup_output_paths
parse_resume_arg "$@"
write_launch_header
start_training
print_status
