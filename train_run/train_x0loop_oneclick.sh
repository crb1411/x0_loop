#!/usr/bin/env bash
# set -euo pipefail
cd /data/seek/aigc/x0_loop
export CUDA_VISIBLE_DEVICES=0
export X0LOOP_RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=40028 \
  -m x0loop.train \
  --config x0loop/configs/default.yaml \
  --runtime-config x0loop/configs/runtime/fsdp_checkpoint_compile.yaml
