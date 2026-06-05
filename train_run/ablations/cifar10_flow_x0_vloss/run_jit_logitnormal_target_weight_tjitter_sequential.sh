#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/run_jit_logitnormal_target_weight_tjitter_sequential.sh [extra x0loop.train args...]
#
# Runs JiT x0-target v-loss target-weight logit-normal mean=0/std=1
# time-condition jitter ablations one by one.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

JITTERS=(
  "0.0 0.01"
  "0.0 0.02"
  "0.01 0.02"
  "0.02 0.02"
  "0.03 0.03"
)

value_tag() {
  printf '%s\n' "$1" | sed 's/-/m/g; s/\./p/g'
}

extract_field() {
  local key="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v key="${key}" '$0 ~ key {print $NF; exit}'
}

launch_one() {
  local mean="$1"
  local std="$2"
  shift 2
  local mean_tag
  local std_tag
  mean_tag="$(value_tag "${mean}")"
  std_tag="$(value_tag "${std}")"

  (
    MODEL_NAME="jit"
    ABLATION_TAG="logitnormal_target_weight_tjitter_mu${mean_tag}_std${std_tag}"
    CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
    EXTRA_SETS=(
      --set "loss.outer_weight=target"
      --set "loss.outer_weight_power=1.0"
      --set "loss.outer_weight_floor=0.0"
      --set "time_sampler.name=logit_normal"
      --set "time_sampler.mean=0.0"
      --set "time_sampler.std=1.0"
      --set "time_condition_jitter.enabled=true"
      --set "time_condition_jitter.mean=${mean}"
      --set "time_condition_jitter.std=${std}"
      --set "time_condition_jitter.prob=1.0"
      --set "time_condition_jitter.min_t=1e-5"
      --set "time_condition_jitter.max_t=0.99999"
    )
    source "${DIR}/_run_ablation.sh" "$@"
  )
}

run_one() {
  local mean="$1"
  local std="$2"
  shift 2
  local mean_tag
  local std_tag
  mean_tag="$(value_tag "${mean}")"
  std_tag="$(value_tag "${std}")"
  local script_name="jit_logitnormal_target_weight_tjitter_mu${mean_tag}_std${std_tag}"

  echo "============================================================"
  echo "[ablation-suite] start ${script_name}"
  local output
  output="$(launch_one "${mean}" "${std}" "$@")"
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

echo "[ablation-suite] total experiments: ${#JITTERS[@]}"
echo "[ablation-suite] extra args: $*"
printf '[ablation-suite] jitters:\n'
printf '  mean std = %s\n' "${JITTERS[@]}"

for pair in "${JITTERS[@]}"; do
  read -r mean std <<< "${pair}"
  run_one "${mean}" "${std}" "$@"
done

echo "============================================================"
echo "[ablation-suite] all experiments completed"
