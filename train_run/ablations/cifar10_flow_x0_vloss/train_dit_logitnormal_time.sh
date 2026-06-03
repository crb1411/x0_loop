#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_logitnormal_time.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="dit"
ABLATION_TAG="logitnormal_time"
CONFIG_PATH="train_run/configs/cifar10/cifar10_dit_flow_train_x0_vloss_heun.yaml"
EXTRA_SETS=(
  --set "loss.outer_weight=none"
  --set "time_sampler.name=logit_normal"
  --set "time_sampler.mean=0.0"
  --set "time_sampler.std=1.0"
)

source "${DIR}/_run_ablation.sh" "$@"
