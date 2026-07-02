#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_SAMPLES=50000
BATCH_SIZE=256
STEPS=20
SAMPLER=heun
CFG=2.2

[ "$#" -eq 1 ] || { echo "usage: $0 /path/to/ckpt_step_XXXXXXXX.pt"; exit 2; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="$(readlink -f "$1")"
[ -f "$CKPT" ] || { echo "[geneval] checkpoint not found: $CKPT"; exit 1; }

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OUT="$(dirname "$CKPT")/geneval_$(basename "$CKPT" .pt)"
LOG="$OUT/geneval.log"
CFG_YAML="$OUT/geneval.yaml"
FAKE_DIR="$OUT/gen_eval/geneval/fake"
mkdir -p "$OUT"

cat > "$CFG_YAML" <<YAML
gen_eval:
  num_samples: $NUM_SAMPLES
  batch_size: $BATCH_SIZE
  steps: $STEPS
  sampler: $SAMPLER
  guidance_scale: $CFG
  guidance_schedule: null
  input2: cifar10-train
  datasets_root: /root/data/cifar10_data
  datasets_download: false
  keep_images: false
  metrics: {isc: true, fid: true, kid: true, ppl: false, prc: true, mind: true}
YAML

echo "[geneval] ckpt: $CKPT"
echo "[geneval] out : $OUT"
echo "[geneval] log : $LOG"

python -m x0loop.eval_fid \
  --ckpt "$CKPT" \
  --eval-config "$CFG_YAML" \
  --set "logging.out_dir=$OUT" \
  --tag geneval \
  > "$LOG" 2>&1 &

echo "[geneval] pid : $!"
echo "[geneval] tail: tail -f $LOG"
echo "[geneval] prog: ls \"$FAKE_DIR\" | wc -l"
