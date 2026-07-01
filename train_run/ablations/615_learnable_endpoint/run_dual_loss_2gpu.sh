#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

EXPERIMENTS=(
  "base_mse_dual_v_b8_c0p01"
  "base_mse_dual_v_b8_c0p05"
  "base_mse_dual_v_b8_c0p1"
  "base_mse_dual_v_b4_c0p05"
)

find_free_port() {
  python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

run_one() {
  local exp="$1"
  local gpu="$2"
  local port
  port="$(find_free_port)"
  echo "[dual-loss] start ${exp} on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  NPROC_PER_NODE=1 \
  MASTER_PORT="${port}" \
  X0LOOP_RUN_BACKGROUND=0 \
    bash "train_run/ablations/615_learnable_endpoint/${exp}/run.sh"
  echo "[dual-loss] done ${exp} on GPU ${gpu}"
}

for ((i = 0; i < ${#EXPERIMENTS[@]}; i += 2)); do
  pids=()
  run_one "${EXPERIMENTS[$i]}" 0 &
  pids+=("$!")

  if (( i + 1 < ${#EXPERIMENTS[@]} )); then
    run_one "${EXPERIMENTS[$((i + 1))]}" 1 &
    pids+=("$!")
  fi

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

echo "[dual-loss] all done"
