#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/run_jit_logitnormal_target_weight_gan_sequential.sh [extra x0loop.train args...]
#
# Runs JiT x0-target v-loss target-weight logit-normal GAN ablations one by one.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
  "gan_late_w0p01_piecewise 0.01 piecewise default"
  "gan_late_w0p02_piecewise 0.02 piecewise default"
  "gan_late_w0p05_piecewise 0.05 piecewise default"
  "gan_late_w0p02_lowt 0.02 piecewise lowt"
  "gan_late_w0p02_tjitter_std0p02 0.02 piecewise tjitter_std0p02"
)

extract_field() {
  local key="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v key="${key}" '$0 ~ key {print $NF; exit}'
}

append_t_weight_sets() {
  local mode="$1"
  case "${mode}" in
    default)
      EXTRA_SETS+=(--set "adversarial.t_weight.name=piecewise")
      ;;
    lowt)
      EXTRA_SETS+=(--set "adversarial.t_weight.name=piecewise")
      EXTRA_SETS+=(--set "adversarial.t_weight.bins=[[0.00,0.05,0.0],[0.05,0.35,1.0],[0.35,1.00,0.0]]")
      ;;
    tjitter_std0p02)
      EXTRA_SETS+=(--set "adversarial.t_weight.name=piecewise")
      EXTRA_SETS+=(--set "time_condition_jitter.enabled=true")
      EXTRA_SETS+=(--set "time_condition_jitter.mean=0.0")
      EXTRA_SETS+=(--set "time_condition_jitter.std=0.02")
      EXTRA_SETS+=(--set "time_condition_jitter.prob=1.0")
      EXTRA_SETS+=(--set "time_condition_jitter.min_t=1e-5")
      EXTRA_SETS+=(--set "time_condition_jitter.max_t=0.99999")
      ;;
    *)
      echo "[ablation-suite] unknown t-weight mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

launch_one() {
  local tag="$1"
  local weight="$2"
  local mode="$4"
  shift 4

  (
    MODEL_NAME="jit"
    ABLATION_TAG="logitnormal_target_weight_${tag}"
    CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
    EXTRA_SETS=(
      --set "loss.outer_weight=target"
      --set "loss.outer_weight_power=1.0"
      --set "loss.outer_weight_floor=0.0"
      --set "time_sampler.name=logit_normal"
      --set "time_sampler.mean=0.0"
      --set "time_sampler.std=1.0"
      --set "adversarial.enabled=true"
      --set "adversarial.fake_space=x0_hat"
      --set "adversarial.loss=hinge"
      --set "adversarial.weight=${weight}"
      --set "adversarial.start_step=30000"
      --set "adversarial.warmup_steps=10000"
      --set "adversarial.update_every=1"
      --set "adversarial.d_steps=1"
      --set "adversarial.clamp_fake_for_d=false"
      --set "adversarial.r1.gamma=1.0"
      --set "adversarial.r1.interval=16"
      --set "discriminator.name=x0_resnet"
      --set "discriminator.base_channels=16"
      --set "discriminator.spectral_norm=true"
      --set "discriminator.time_projection=true"
      --set "discriminator.class_projection=true"
      --set "discriminator.lr=2e-4"
      --set "discriminator.weight_decay=0.0"
      --set "discriminator.betas=[0.0,0.99]"
    )
    append_t_weight_sets "${mode}"
    source "${DIR}/_run_ablation.sh" "$@"
  )
}

run_one() {
  local tag="$1"
  local weight="$2"
  local t_weight_name="$3"
  local mode="$4"
  shift 4
  local script_name="jit_logitnormal_target_weight_${tag}"

  echo "============================================================"
  echo "[ablation-suite] start ${script_name} weight=${weight} t_weight=${t_weight_name}/${mode}"
  local output
  output="$(launch_one "${tag}" "${weight}" "${t_weight_name}" "${mode}" "$@")"
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

echo "[ablation-suite] total experiments: ${#EXPERIMENTS[@]}"
echo "[ablation-suite] extra args: $*"
printf '[ablation-suite] experiments:\n'
printf '  %s\n' "${EXPERIMENTS[@]}"

for experiment in "${EXPERIMENTS[@]}"; do
  read -r tag weight t_weight_name mode <<< "${experiment}"
  run_one "${tag}" "${weight}" "${t_weight_name}" "${mode}" "$@"
done

echo "============================================================"
echo "[ablation-suite] all experiments completed"
