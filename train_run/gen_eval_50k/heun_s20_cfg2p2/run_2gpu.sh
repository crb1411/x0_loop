#!/usr/bin/env bash
# Run a 50k FID/IS/KID eval for heun, steps=20, cfg=2.2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

EXP="heun_s20_cfg2p2_50k"
CKPT="/data/seek/aigc/x0_loop/runs/ablations/cifar10_flow_x0_vloss/jit/learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
EVAL_CFG="${ROOT}/train_run/gen_eval_50k/heun_s20_cfg2p2/eval.yaml"
OUT_DIR="/data/seek/aigc/x0_loop/runs/gen_eval_50k/heun_s20_cfg2p2"
LOG_DIR="${OUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/eval.log"
PYTHON="${PYTHON:-/root/miniconda3/envs/vl/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/vl/bin/torchrun}"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "[gen_eval_50k] exp=${EXP} sampler=heun steps=20 cfg=2.2 samples=50000 GPUs=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "[gen_eval_50k] python=${PYTHON}" | tee -a "${LOG_FILE}"
echo "[gen_eval_50k] torchrun=${TORCHRUN}" | tee -a "${LOG_FILE}"
echo "[gen_eval_50k] out=${OUT_DIR}" | tee -a "${LOG_FILE}"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=2 \
  -m x0loop.eval_fid \
  --ckpt "${CKPT}" \
  --eval-config "${EVAL_CFG}" \
  --tag "${EXP}" \
  --set "logging.out_dir=${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "[gen_eval_50k] done -> ${OUT_DIR}/gen_eval_metrics_*.jsonl" | tee -a "${LOG_FILE}"
