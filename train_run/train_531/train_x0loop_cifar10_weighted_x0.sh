#!/usr/bin/env bash
cd /data/seek/aigc/x0_loop

source train_run/common_torchrun.sh
setup_torchrun_env

RESUME_ARGS=""
if [ -n "$1" ]; then
  RESUME_ARGS="--set train.resume=$1"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m x0loop.train \
  --config train_run/configs/cifar10/cifar10_dit_flow_train_x0_weighted.yaml \
  --runtime-config x0loop/configs/runtime/fsdp_checkpoint_compile.yaml \
  $RESUME_ARGS
