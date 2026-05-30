#!/usr/bin/env bash
# Usage:
#   bash infer/infer_cifar10.sh                               # 使用脚本内指定的 checkpoint
#   bash infer/infer_cifar10.sh /path/to/ckpt_step_xxx.pt    # 手动指定 checkpoint
#
# Base config 自动读取 <训练目录>/resolved_config.yaml
# Infer 参数覆盖来自 infer/infer.yaml
#
# 输出默认位置：<checkpoint 所在目录>/infer_step_XXXXXXXX/
#   step_XXXXXXXX_sample_NNN_x0loop.png   — 每张图的去噪轨迹
#   sample_grid.png                        — 所有样本的平铺网格
#   infer_config.yaml                      — 本次推理的完整配置

cd /data/seek/aigc/x0_loop
export CUDA_VISIBLE_DEVICES=0

# ─── 在这里配置 checkpoint 路径 ───────────────────────────────────────────
CKPT="/data/seek/aigc/x0_loop/runs/cifar10_diffusion/20260530_141408_train/checkpoints/ckpt_step_00148000.pt"
# ─────────────────────────────────────────────────────────────────────────

# 允许命令行传入覆盖
if [ -n "$1" ]; then
  CKPT="$1"
fi

if [ ! -f "$CKPT" ]; then
  echo "[error] checkpoint not found: $CKPT"
  exit 1
fi

python -m x0loop.infer \
  --ckpt         "$CKPT" \
  --infer-config infer/infer.yaml
