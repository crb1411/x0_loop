#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

EXPERIMENTS=(
  "scheme2_beta0p2"
  "scheme2_beta0p5"
  "scheme2_beta0p8"
  "scheme1_beta0p2"
  "scheme1_beta0p5"
  "scheme1_beta0p8"
)

run_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  local script="${ROOT}/train_run/ablations/615_learnable_endpoint/${name}/run.sh"
  CUDA_VISIBLE_DEVICES="${gpu}" X0LOOP_RUN_BACKGROUND=0 bash "${script}" "$@"
}

wait_pair() {
  local pid_a="$1"
  local name_a="$2"
  local pid_b="${3:-}"
  local name_b="${4:-}"
  local failed=0

  if ! wait "${pid_a}"; then
    echo "[ablation-suite] failed: ${name_a}" >&2
    failed=1
  fi

  if [ -n "${pid_b}" ]; then
    if ! wait "${pid_b}"; then
      echo "[ablation-suite] failed: ${name_b}" >&2
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
