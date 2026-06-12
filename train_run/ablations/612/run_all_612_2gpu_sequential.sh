#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/612/run_all_612_2gpu_sequential.sh [extra x0loop.train args...]
#
# Runs the 2026-06-12 JiT 2-GPU ablations one by one.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/_common_jit_tjitter_2gpu.sh"

EXPERIMENTS=(
  "jit_tjitter_mu0p02_std0p02_baseline 0.02 0.02 none"
  "jit_tjitter_mu0p015_std0p02 0.015 0.02 none"
  "jit_tjitter_mu0p02_std0p015 0.02 0.015 none"
  "jit_tjitter_mu0p025_std0p02 0.025 0.02 none"
  "jit_tjitter_mu0p02_std0p025 0.02 0.025 none"
  "jit_tjitter_mu0p02_std0p02_gan_nohigh_w0p005 0.02 0.02 0.005"
  "jit_tjitter_mu0p02_std0p02_gan_nohigh_w0p01 0.02 0.02 0.01"
)

extract_field() {
  local key="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v key="${key}" '$0 ~ key {print $NF; exit}'
}

launch_one() {
  local tag_suffix="$1"
  local jitter_mean="$2"
  local jitter_std="$3"
  local gan_weight="$4"
  shift 4

  (
    TAG_SUFFIX="${tag_suffix}"
    JITTER_MEAN="${jitter_mean}"
    JITTER_STD="${jitter_std}"
    if [ "${gan_weight}" != "none" ]; then
      GAN_NOHIGH_WEIGHT="${gan_weight}"
    else
      unset GAN_NOHIGH_WEIGHT || true
    fi
    launch_jit_tjitter_2gpu "$@"
  )
}

run_one() {
  local tag_suffix="$1"
  local jitter_mean="$2"
  local jitter_std="$3"
  local gan_weight="$4"
  shift 4

  echo "============================================================"
  echo "[ablation-suite] start ${tag_suffix} jitter=${jitter_mean}/${jitter_std} gan_nohigh=${gan_weight}"
  local output
  output="$(launch_one "${tag_suffix}" "${jitter_mean}" "${jitter_std}" "${gan_weight}" "$@")"
  printf '%s\n' "${output}"

  local pid
  local log_file
  pid="$(extract_field "\\[x0loop\\] pid:" "${output}")"
  log_file="$(extract_field "\\[x0loop\\] log:" "${output}")"

  if [ -z "${pid}" ]; then
    echo "[ablation-suite] failed to parse PID for ${tag_suffix}" >&2
    exit 1
  fi

  echo "[ablation-suite] waiting for ${tag_suffix}, pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 10
  done

  local status
  status="unknown"
  if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
    status="$(awk '/\[x0loop\] train exited with status / {s=$NF} END {print s}' "${log_file}")"
  fi

  if [ "${status}" != "0" ]; then
    echo "[ablation-suite] ${tag_suffix} failed or did not report success, status=${status}" >&2
    if [ -n "${log_file}" ]; then
      echo "[ablation-suite] log: ${log_file}" >&2
    fi
    exit 1
  fi

  echo "[ablation-suite] done ${tag_suffix}"
  if [ -n "${log_file}" ]; then
    echo "[ablation-suite] log: ${log_file}"
  fi
}

echo "[ablation-suite] total experiments: ${#EXPERIMENTS[@]}"
echo "[ablation-suite] default CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}"
echo "[ablation-suite] default NPROC_PER_NODE=${NPROC_PER_NODE:-2}"
echo "[ablation-suite] extra args: $*"
printf '[ablation-suite] experiments:\n'
printf '  %s\n' "${EXPERIMENTS[@]}"

for experiment in "${EXPERIMENTS[@]}"; do
  read -r tag_suffix jitter_mean jitter_std gan_weight <<< "${experiment}"
  run_one "${tag_suffix}" "${jitter_mean}" "${jitter_std}" "${gan_weight}" "$@"
done

echo "============================================================"
echo "[ablation-suite] all experiments completed"
