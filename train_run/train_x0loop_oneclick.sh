#!/usr/bin/env bash
# set -euo pipefail
cd /data/seek/aigc/x0_loop

source train_run/common_torchrun.sh
setup_torchrun_env

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config x0loop/configs/default.yaml \
  --runtime-config x0loop/configs/runtime/fsdp_checkpoint_compile.yaml
