#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/train_dit_x0_weight.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="dit"
ABLATION_TAG="x0_weight"
CONFIG_PATH="train_run/configs/cifar10/cifar10_dit_flow_train_x0_vloss_heun.yaml"
EXTRA_SETS=(
  --set "loss.outer_weight=x0"
  --set "loss.outer_weight_power=2.0"
  --set "loss.outer_weight_floor=0.0"
  --set "time_sampler.name=uniform_continuous"
)

source "${DIR}/_run_ablation.sh" "$@"
