#!/usr/bin/env bash
# Shared 2-GPU JiT launcher for the 2026-06-12 CIFAR10 ablations.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

append_gan_nohigh_sets() {
  local weight="$1"
  EXTRA_SETS+=(
    --set "adversarial.enabled=true"
    --set "adversarial.fake_space=x0_hat"
    --set "adversarial.loss=hinge"
    --set "adversarial.weight=${weight}"
    --set "adversarial.start_step=30000"
    --set "adversarial.warmup_steps=10000"
    --set "adversarial.update_every=1"
    --set "adversarial.d_steps=1"
    --set "adversarial.clamp_fake_for_d=false"
    --set "adversarial.t_weight.name=piecewise"
    --set "adversarial.t_weight.bins=[[0.00,0.05,0.25],[0.05,0.35,1.0],[0.35,0.65,0.5],[0.65,1.00,0.0]]"
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
}

launch_jit_tjitter_2gpu() {
  : "${TAG_SUFFIX:?TAG_SUFFIX is required}"
  : "${JITTER_MEAN:?JITTER_MEAN is required}"
  : "${JITTER_STD:?JITTER_STD is required}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
  export RUNTIME_CONFIG="${RUNTIME_CONFIG:-x0loop/configs/runtime/ddp_checkpoint_compile.yaml}"

  MODEL_NAME="jit"
  ABLATION_TAG="612_2gpu_${TAG_SUFFIX}"
  CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
  EXTRA_SETS=(
    --set "train.batch_size=256"
    --set "eval.batch_size=256"
    --set "loss.outer_weight=target"
    --set "loss.outer_weight_power=1.0"
    --set "loss.outer_weight_floor=0.0"
    --set "time_sampler.name=logit_normal"
    --set "time_sampler.mean=0.0"
    --set "time_sampler.std=1.0"
    --set "time_condition_jitter.enabled=true"
    --set "time_condition_jitter.mean=${JITTER_MEAN}"
    --set "time_condition_jitter.std=${JITTER_STD}"
    --set "time_condition_jitter.prob=1.0"
    --set "time_condition_jitter.min_t=1e-5"
    --set "time_condition_jitter.max_t=0.99999"
  )

  if [ -n "${GAN_NOHIGH_WEIGHT:-}" ]; then
    append_gan_nohigh_sets "${GAN_NOHIGH_WEIGHT}"
  fi

  source "${DIR}/../cifar10_flow_x0_vloss/_run_ablation.sh" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  launch_jit_tjitter_2gpu "$@"
fi
