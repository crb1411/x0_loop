#!/usr/bin/env bash
# Usage:
#   bash train_run/ablations/cifar10_flow_x0_vloss/train_jit_v_weight.sh [ckpt.pt] [extra x0loop.train args...]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="jit"
ABLATION_TAG="v_weight"
CONFIG_PATH="train_run/configs/cifar10/cifar10_jit_flow_train_x0_vloss_heun.yaml"
EXTRA_SETS=(
  --set "loss.outer_weight=target"
  --set "loss.outer_weight_power=1.0"
  --set "loss.outer_weight_floor=0.0"
  --set "time_sampler.name=uniform_continuous"
)

source "${DIR}/_run_ablation.sh" "$@"
