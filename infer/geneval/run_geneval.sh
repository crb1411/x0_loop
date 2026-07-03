#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STEPS="${STEPS:-10}"
SAMPLER="${SAMPLER:-clean_loop}"
CFG="${CFG:-2.2}"
REFINE_TIME="${REFINE_TIME:-0.5}"
GUIDANCE_SCHEDULE="${GUIDANCE_SCHEDULE:-null}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"
PYTHON_CMD="${PYTHON_CMD:-/root/miniconda3/envs/vl/bin/python}"

[ "$#" -eq 1 ] || { echo "usage: $0 /path/to/ckpt_step_XXXXXXXX.pt"; exit 2; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="$(readlink -f "$1")"
[ -f "$CKPT" ] || { echo "[geneval] checkpoint not found: $CKPT"; exit 1; }

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/p/g; s/^p|p$//g'; }

RUN_TAG="${RUN_TAG:-$(slug "${SAMPLER}_s${STEPS}_cfg${CFG}_rt${REFINE_TIME}")}"
OUT="${GENEVAL_OUT:-$(dirname "$CKPT")/geneval_$(basename "$CKPT" .pt)_${RUN_TAG}}"
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
  guidance_schedule: $GUIDANCE_SCHEDULE
  refine_time: $REFINE_TIME
  input2: cifar10-train
  datasets_root: /root/data/cifar10_data
  datasets_download: false
  keep_images: false
  metrics: {isc: true, fid: true, kid: true, ppl: false, prc: true, mind: true}
YAML

echo "[geneval] ckpt: $CKPT"
echo "[geneval] out : $OUT"
echo "[geneval] log : $LOG"
echo "[geneval] cfg : sampler=$SAMPLER steps=$STEPS guidance_scale=$CFG guidance_schedule=$GUIDANCE_SCHEDULE refine_time=$REFINE_TIME"

read -r -a python_cmd <<< "$PYTHON_CMD"
cmd=("${python_cmd[@]}" -m x0loop.eval_fid
  --ckpt "$CKPT"
  --eval-config "$CFG_YAML"
  --set "logging.out_dir=$OUT"
  --tag geneval)

if [ "$RUN_FOREGROUND" = "1" ]; then
  "${cmd[@]}" > "$LOG" 2>&1
  echo "[geneval] done: $LOG"
  exit $?
fi

nohup "${cmd[@]}" > "$LOG" 2>&1 &
echo "[geneval] pid : $!"
echo "[geneval] tail: tail -f $LOG"
echo "[geneval] prog: ls \"$FAKE_DIR\" 2>/dev/null | wc -l"
