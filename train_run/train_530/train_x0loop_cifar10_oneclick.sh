#!/usr/bin/env bash
# set -euo pipefail
# Usage:
#   bash train_run/train_x0loop_cifar10_oneclick.sh                          # 从头训练
#   bash train_run/train_x0loop_cifar10_oneclick.sh /path/to/ckpt_step_xxx.pt  # 断点续训
cd /data/seek/aigc/x0_loop
export CUDA_VISIBLE_DEVICES=0
export X0LOOP_RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

RESUME_ARGS=""
if [ -n "$1" ]; then
  RESUME_ARGS="--set train.resume=$1"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=40030 \
  -m x0loop.train \
  --config train_run/configs/cifar10/cifar10_dit_flow_train.yaml \
  --runtime-config x0loop/configs/runtime/fsdp_checkpoint_compile.yaml \
  $RESUME_ARGS
