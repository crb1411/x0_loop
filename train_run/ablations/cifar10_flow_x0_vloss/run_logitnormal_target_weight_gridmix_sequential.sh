#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/run_logitnormal_target_weight_gridmix_sequential.sh [extra x0loop.train args...]
#
# Runs target-weight logit-normal mean=0/std=1 ablations for DiT and JiT:
#   no grid mixing, grid_mix_prob=0.5, grid_mix_prob=0.3, grid_mix_prob=0.8.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

SCRIPTS=(
  train_dit_logitnormal_target_weight_gridmix_p0p5.sh
  train_dit_logitnormal_target_weight_gridmix_p0p3.sh
  train_dit_logitnormal_target_weight_gridmix_p0p8.sh
  train_dit_logitnormal_target_weight.sh
  train_jit_logitnormal_target_weight_gridmix_p0p5.sh
  train_jit_logitnormal_target_weight_gridmix_p0p3.sh
  train_jit_logitnormal_target_weight_gridmix_p0p8.sh
  train_jit_logitnormal_target_weight.sh
)

extract_field() {
  local key="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v key="${key}" '$0 ~ key {print $NF; exit}'
}

run_one() {
  local script_name="$1"
  shift

  echo "============================================================"
  echo "[ablation-suite] start ${script_name}"
  local output
  output="$(bash "${DIR}/${script_name}" "$@")"
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

echo "[ablation-suite] total experiments: ${#SCRIPTS[@]}"
echo "[ablation-suite] extra args: $*"
printf '[ablation-suite] scripts:\n'
printf '  %s\n' "${SCRIPTS[@]}"

for script_name in "${SCRIPTS[@]}"; do
  run_one "${script_name}" "$@"
done

echo "============================================================"
echo "[ablation-suite] all experiments completed"
