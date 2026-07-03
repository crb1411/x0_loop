#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

GPU0_EXPERIMENTS=(
  "resample_local_tbank0p75_w2"
  "resample_sampler_tbank0p75_w2"
  "resample_local_tbank0p50_w2"
)

GPU1_EXPERIMENTS=(
  "step_local_tbank0p75_w2"
  "step_sampler_tbank0p75_w2"
  "resample_local_tbank0p75_w1"
)

find_free_port() {
  python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

run_queue() {
  local gpu="$1"
  shift
  local exp port script
  for exp in "$@"; do
    port="$(find_free_port)"
    script="${ROOT}/train_run2/x0loop_bank_ablation/${exp}/run.sh"
    echo "[x0loop-bank] start ${exp} on GPU ${gpu} port ${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    NPROC_PER_NODE=1 \
    MASTER_PORT="${port}" \
    X0LOOP_RUN_BACKGROUND=0 \
      bash "${script}"
    echo "[x0loop-bank] done  ${exp} on GPU ${gpu}"
  done
}

run_queue 0 "${GPU0_EXPERIMENTS[@]}" &
pid0=$!

run_queue 1 "${GPU1_EXPERIMENTS[@]}" &
pid1=$!

failed=0
if ! wait "${pid0}"; then
  echo "[x0loop-bank] GPU0 queue failed" >&2
  failed=1
fi
if ! wait "${pid1}"; then
  echo "[x0loop-bank] GPU1 queue failed" >&2
  failed=1
fi

if [ "${failed}" -ne 0 ]; then
  exit 1
fi

echo "[x0loop-bank] all done"
