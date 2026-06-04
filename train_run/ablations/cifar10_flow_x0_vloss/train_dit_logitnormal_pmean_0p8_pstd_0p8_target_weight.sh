#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_logitnormal_pmean_0p8_pstd_0p8_target_weight.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="dit"
ABLATION_TAG="logitnormal_pmean_0p8_pstd_0p8_target_weight"
CONFIG_PATH="train_run/configs/cifar10/cifar10_dit_flow_train_x0_vloss_heun.yaml"
EXTRA_SETS=(
  --set "loss.outer_weight=target"
  --set "loss.outer_weight_power=1.0"
  --set "loss.outer_weight_floor=0.0"
  --set "time_sampler.name=logit_normal"
  --set "time_sampler.mean=0.8"
  --set "time_sampler.std=0.8"
)

source "${DIR}/_run_ablation.sh" "$@"
