#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_logitnormal_target_weight_gridmix_p0p5.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="jit"
ABLATION_TAG="logitnormal_target_weight_gridmix_p0p5"
CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
EXTRA_SETS=(
  --set "loss.outer_weight=target"
  --set "loss.outer_weight_power=1.0"
  --set "loss.outer_weight_floor=0.0"
  --set "time_sampler.name=logit_normal"
  --set "time_sampler.mean=0.0"
  --set "time_sampler.std=1.0"
  --set "time_sampler.grid_mix_prob=0.5"
  --set "time_sampler.grid_steps=[50,20]"
)

source "${DIR}/_run_ablation.sh" "$@"
