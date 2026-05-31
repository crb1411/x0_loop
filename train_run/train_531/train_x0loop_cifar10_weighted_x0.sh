#!/usr/bin/env bash
cd /data/seek/aigc/x0_loop

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export X0LOOP_RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

echo "[torchrun] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
