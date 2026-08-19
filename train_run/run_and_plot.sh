#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <log_file> <timestamp> <root> -- <command...>" >&2
  exit 2
fi

LOG_FILE="$1"
X0LOOP_RUN_TIMESTAMP="$2"
ROOT="$3"
shift 3

if [ "${1:-}" = "--" ]; then
  shift
fi

set +e
if command -v stdbuf >/dev/null 2>&1; then
  PYTHONUNBUFFERED=1 stdbuf -oL -eL "$@" 2>&1 | tee -a "${LOG_FILE}"
  TRAIN_STATUS=${PIPESTATUS[0]}
else
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "${LOG_FILE}"
  TRAIN_STATUS=${PIPESTATUS[0]}
fi
set -e

{
  echo "[x0loop] train exited with status ${TRAIN_STATUS}"
  if [ "${TRAIN_STATUS}" -eq 0 ]; then
    METRICS_FILE="$(grep -o "metrics_file=[^ ]*" "${LOG_FILE}" | tail -n 1 | cut -d= -f2- || true)"
    if [ -z "${METRICS_FILE}" ] || [ ! -f "${METRICS_FILE}" ]; then
      METRICS_FILE="$(
        find "${ROOT}/runs" -name "metrics_${X0LOOP_RUN_TIMESTAMP}*.jsonl" -type f -printf "%T@ %p\n" 2>/dev/null \
          | sort -nr \
          | head -n 1 \
          | cut -d" " -f2- || true
      )"
    fi

    if [ -n "${METRICS_FILE}" ] && [ -f "${METRICS_FILE}" ]; then
      echo "[x0loop] plotting metrics: ${METRICS_FILE}"
      set +e
      uv run python "${ROOT}/tools/plot_training_trends.py" "${METRICS_FILE}"
      PLOT_STATUS=$?
      set -e
      if [ "${PLOT_STATUS}" -ne 0 ]; then
        echo "[x0loop] plotting failed with status ${PLOT_STATUS}; training result is still status ${TRAIN_STATUS}"
      fi
    else
      echo "[x0loop] no metrics jsonl found for timestamp ${X0LOOP_RUN_TIMESTAMP}; skip plotting"
    fi
  else
    echo "[x0loop] training failed or was interrupted; skip plotting"
  fi
} 2>&1 | tee -a "${LOG_FILE}"

exit "${TRAIN_STATUS}"
