#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

EXPERIMENTS=(
  "base2"
  "base2_infer_tshift_p0p01"
  "base2_infer_tshift_p0p02"
  "base2_infer_tshift_p0p03"
  "base2_infer_tshift_m0p02"
  "base2_tjitter_mu0p02_std0p02"
  "base2_tjitter_mu0p00_std0p02"
)

find_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

run_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  local script="${ROOT}/train_run/ablations/615_learnable_endpoint/${name}/run.sh"
  local port
  port="$(find_free_port)"
  CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT="${port}" X0LOOP_RUN_BACKGROUND=0 bash "${script}" "$@"
}

wait_pair() {
  local pid_a="$1"
  local name_a="$2"
  local pid_b="${3:-}"
  local name_b="${4:-}"
  local failed=0

  if ! wait "${pid_a}"; then
    echo "[time-condition-shift] failed: ${name_a}" >&2
    failed=1
  fi

  if [ -n "${pid_b}" ]; then
    if ! wait "${pid_b}"; then
      echo "[time-condition-shift] failed: ${name_b}" >&2
      failed=1
    fi
  fi

  if [ "${failed}" -ne 0 ]; then
    exit 1
  fi
}

idx=0
while [ "${idx}" -lt "${#EXPERIMENTS[@]}" ]; do
  name0="${EXPERIMENTS[$idx]}"
  run_one "${name0}" 0 "$@" &
  pid0=$!
  idx=$((idx + 1))

  pid1=""
  name1=""
  if [ "${idx}" -lt "${#EXPERIMENTS[@]}" ]; then
    name1="${EXPERIMENTS[$idx]}"
    run_one "${name1}" 1 "$@" &
    pid1=$!
    idx=$((idx + 1))
  fi

  wait_pair "${pid0}" "${name0}" "${pid1}" "${name1}"
done
