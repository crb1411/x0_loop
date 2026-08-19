#!/usr/bin/env bash
# Auto-generated, fully self-contained single-GPU FID eval for one v2_616 experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${ROOT}"

EXP="dpmpp_2m_s20_cfg2"
CKPT="./runs/ablations/cifar10_flow_x0_vloss/jit/learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
EVAL_CFG="${ROOT}/train_run/sampler_ablation/v2_616/${EXP}/eval.yaml"
OUT_DIR="./runs/sampler_ablation/v2_616/${EXP}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval.log"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[sampler_ablation_v2_616] exp=${EXP} sampler=dpmpp_2m steps=20 cfg=2.0 GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "[sampler_ablation_v2_616] out=${OUT_DIR}" | tee -a "${LOG_FILE}"

uv run python -m x0loop.eval_fid \
  --ckpt "${CKPT}" \
  --eval-config "${EVAL_CFG}" \
  --tag "${EXP}" \
  --set "logging.out_dir=${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "[sampler_ablation_v2_616] done -> ${OUT_DIR}/gen_eval_metrics_*.jsonl" | tee -a "${LOG_FILE}"
