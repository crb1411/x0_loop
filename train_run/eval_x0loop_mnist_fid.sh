#!/usr/bin/env bash
# set -euo pipefail
cd /data/seek/aigc/x0_loop
export CUDA_VISIBLE_DEVICES=0
export X0LOOP_RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

python -m x0loop.eval_mnist_fid \
  --config x0loop/configs/mnist_dit_diffusion.yaml \
  --runtime-config x0loop/configs/runtime/fsdp_checkpoint_compile.yaml \
  --ckpt /data/seek/aigc/x0_loop/runs/mnist_diffusion/run_20260213_232141/checkpoints/ckpt_step_03280000.pt \
  --out-dir-base /data/seek/aigc/x0_loop/runs/mnist_diffusion/fid_mnist \
  --num-samples 10000 \
  --steps 100 \
  --sampler auto
