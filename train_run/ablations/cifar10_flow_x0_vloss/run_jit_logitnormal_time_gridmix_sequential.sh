#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/run_jit_logitnormal_time_gridmix_sequential.sh [extra x0loop.train args...]
#
# Runs JiT unweighted logit-normal mean=0/std=1 grid-mix ablations one by one:
#   grid_mix_prob=0.3, grid_mix_prob=0.5, grid_mix_prob=0.8.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

GRID_PROBS=(0.3 0.5 0.8)

prob_tag() {
  printf '%s\n' "$1" | sed 's/\./p/g'
}

extract_field() {
  local key="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v key="${key}" '$0 ~ key {print $NF; exit}'
}

launch_one() {
  local prob="$1"
  shift
  local tag_prob
  tag_prob="$(prob_tag "${prob}")"

  (
    MODEL_NAME="jit"
    ABLATION_TAG="logitnormal_time_gridmix_p${tag_prob}"
    CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
    EXTRA_SETS=(
      --set "loss.outer_weight=none"
      --set "time_sampler.name=logit_normal"
      --set "time_sampler.mean=0.0"
      --set "time_sampler.std=1.0"
      --set "time_sampler.grid_mix_prob=${prob}"
      --set "time_sampler.grid_steps=[50,20]"
    )
    source "${DIR}/_run_ablation.sh" "$@"
  )
}

run_one() {
  local prob="$1"
  shift
  local tag_prob
  tag_prob="$(prob_tag "${prob}")"
  local script_name="jit_logitnormal_time_gridmix_p${tag_prob}"

  echo "============================================================"
  echo "[ablation-suite] start ${script_name}"
  local output
  output="$(launch_one "${prob}" "$@")"
  printf '%s\n' "${output}"

  local pid
  local log_file
  pid="$(extract_field "\\[x0loop\\] pid:" "${output}")"
  log_file="$(extract_field "\\[x0loop\\] log:" "${output}")"

  if [ -z "${pid}" ]; then
    echo "[ablation-suite] failed to parse PID for ${script_name}" >&2
    exit 1
  fi

  echo "[ablation-suite] waiting for ${script_name}, pid=${pid}"
  echo "[ablation-suite] next experiment will start only after training and plotting finish"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 10
  done

  local status
  status="unknown"
  if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
    status="$(awk '/\[x0loop\] train exited with status / {s=$NF} END {print s}' "${log_file}")"
  fi

  if [ "${status}" != "0" ]; then
    echo "[ablation-suite] ${script_name} failed or did not report success, status=${status}" >&2
    if [ -n "${log_file}" ]; then
      echo "[ablation-suite] log: ${log_file}" >&2
    fi
    exit 1
  fi

  echo "[ablation-suite] done ${script_name}"
  if [ -n "${log_file}" ]; then
    echo "[ablation-suite] log: ${log_file}"
  fi
}

echo "[ablation-suite] total experiments: ${#GRID_PROBS[@]}"
echo "[ablation-suite] extra args: $*"
printf '[ablation-suite] grid probs:\n'
printf '  %s\n' "${GRID_PROBS[@]}"

for prob in "${GRID_PROBS[@]}"; do
  run_one "${prob}" "$@"
done

echo "============================================================"
echo "[ablation-suite] all experiments completed"
