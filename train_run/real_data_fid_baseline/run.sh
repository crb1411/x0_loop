#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${DATASET_ROOT:-/root/data/cifar10_data}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
# INPUT2 can be cifar10-train, cifar10-val, or edm-cifar10-32x32.
INPUT2="${INPUT2:-cifar10-train}"
SEED="${SEED:-42}"
TIMESTAMP="${X0LOOP_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/real_data_fid_baseline/${TIMESTAMP}}"
LOG_FILE="${OUT_DIR}/real_data_fid.log"

mkdir -p "${OUT_DIR}"

echo "[real_fid] dataset_root=${DATASET_ROOT}" | tee -a "${LOG_FILE}"
echo "[real_fid] num_samples=${NUM_SAMPLES}" | tee -a "${LOG_FILE}"
echo "[real_fid] input2=${INPUT2}" | tee -a "${LOG_FILE}"
echo "[real_fid] out_dir=${OUT_DIR}" | tee -a "${LOG_FILE}"

python train_run/real_data_fid_baseline/eval_real_cifar10.py \
  --dataset-root "${DATASET_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --input2 "${INPUT2}" \
  --seed "${SEED}" \
  --cuda \
  "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "[real_fid] metrics -> ${OUT_DIR}/real_data_metrics.jsonl" | tee -a "${LOG_FILE}"
