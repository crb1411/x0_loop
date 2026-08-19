#!/usr/bin/env bash
# Auto-generated, fully self-contained single-GPU FID eval for one v3_616 experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${ROOT}"

EXP="heun_s20_cfg1p7"
CKPT="./runs/ablations/cifar10_flow_x0_vloss/jit/learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
EVAL_CFG="${ROOT}/train_run/sampler_ablation/v3_616/${EXP}/eval.yaml"
OUT_DIR="./runs/sampler_ablation/v3_616/${EXP}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval.log"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[sampler_ablation_v3_616] exp=${EXP} sampler=heun steps=20 cfg=1.7 GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "[sampler_ablation_v3_616] out=${OUT_DIR}" | tee -a "${LOG_FILE}"

uv run python -m x0loop.eval_fid \
  --ckpt "${CKPT}" \
  --eval-config "${EVAL_CFG}" \
  --tag "${EXP}" \
  --set "logging.out_dir=${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "[sampler_ablation_v3_616] done -> ${OUT_DIR}/gen_eval_metrics_*.jsonl" | tee -a "${LOG_FILE}"
