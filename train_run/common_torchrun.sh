#!/usr/bin/env bash

# Shared torchrun helpers for local single-node experiments.
# Source this file from any train_run script, then call:
#   setup_x0loop_repo "$0"
#   setup_torchrun_env
#   run_x0loop_train <config.yaml> [checkpoint.pt] [extra --set args...]

find_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

setup_x0loop_repo() {
  local script_path="${1:-$0}"
  local script_dir
  local candidate
  script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
  candidate="${script_dir}"
  while [ "${candidate}" != "/" ]; do
    if [ -d "${candidate}/x0loop" ] && [ -d "${candidate}/train_run" ]; then
      break
    fi
    candidate="$(dirname "${candidate}")"
  done
  if [ ! -d "${candidate}/x0loop" ] || [ ! -d "${candidate}/train_run" ]; then
    echo "[x0loop] failed to locate repository root from ${script_path}" >&2
    return 1
  fi
  export X0LOOP_ROOT="${X0LOOP_ROOT:-${candidate}}"
  cd "${X0LOOP_ROOT}"
  export PYTHONPATH="${X0LOOP_ROOT}:${PYTHONPATH:-}"
  echo "[x0loop] root=${X0LOOP_ROOT}"
}

setup_torchrun_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export X0LOOP_RUN_TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}" 
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-$(find_free_port)}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
}

run_x0loop_train() {
  local config_path="$1"
  shift
  local resume_path=""
  if [ "$#" -gt 0 ] && [[ "${1}" != --* ]]; then
    resume_path="$1"
    shift
  fi

  local resume_args=()
  if [ -n "${resume_path}" ]; then
    resume_args+=(--set "train.resume=${resume_path}")
  fi

  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m x0loop.train \
    --config "${config_path}" \
    --runtime-config "${RUNTIME_CONFIG:-x0loop/configs/runtime/fsdp_checkpoint_compile.yaml}" \
    "${resume_args[@]}" \
    "$@"
}
