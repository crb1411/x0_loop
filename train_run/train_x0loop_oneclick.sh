#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common_torchrun.sh"
setup_x0loop_repo "$0"
setup_torchrun_env

run_x0loop_train \
  x0loop/configs/default.yaml \
  "$@"
