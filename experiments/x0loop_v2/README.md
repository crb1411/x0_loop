# x0loop v2 experiments

All work in this directory follows
[`docs/x0loop_research_principles.md`](../../docs/x0loop_research_principles.md)
v1. Concrete results, invalid runs, stopping decisions, and the next registered
hypothesis live in
[`docs/x0loop_v2_training_analysis.md`](../../docs/x0loop_v2_training_analysis.md).
If this README conflicts with either document, stop and update the documentation
before launching another run.

## Authoritative protocol

The research objective is final CIFAR-10 generation quality. The primary result
is EMA Heun-20 / CFG-2.2 FID over 50,000 generated images with the fixed
reference statistics, seed, and cyclic label order. KID, precision/recall,
NFE 4/8/20, and fixed-root trajectory traces determine robustness and mechanism;
they do not replace authoritative FID.

Formal claims require a random-initialized 300-epoch run with full fresh
exposure and an equal-budget FRESH baseline. A shared checkpoint may be used for
a pre-registered 5k-step screening run, but that result is not independent
from-zero evidence. During a formal 58,500-step run, fixed 5k FID is normally
evaluated only at 15k/30k/45k; only surviving candidates receive final 50k FID.

The only permitted external absolute path is the CIFAR-10 root
`/mnt/data/crb/data`. Commands run through the repository uv environment and
all code, logs, checkpoints, and evaluation outputs use workspace-relative
paths. GPU 7 is preferred, with GPU 6 as fallback.

## Branch semantics

`run_from_scratch.sh` has explicit names so historical definitions remain
reproducible:

| branch | rollout source | auxiliary target |
|---|---|---|
| `fresh` | none | none |
| `bank-fix` | stratified EMA Heun replay | legacy teacher velocity |
| `bank-x0` | stratified EMA Heun replay | teacher native x0 |
| `online` | current EMA online Heun rollout | legacy teacher velocity |
| `online-x0` | current EMA online Heun rollout | teacher native x0 |

All auxiliary branches retain the complete fresh batch. The target output
gradient ratio, auxiliary batch ratio, compile mode, and resume point must be
explicitly registered in the analysis document before launch. Training and FID
both use the correct learnable endpoint, Heun grid, CFG, EMA, and label rules.

Start a 300-epoch from-zero FRESH run:

```bash
X0LOOP_GPU=7 X0LOOP_CYCLE=cycleNN \
  experiments/x0loop_v2/run_from_scratch.sh fresh
```

Example pre-registered 5k-step x0-target screen from a shared step-10k prefix:

```bash
X0LOOP_GPU=7 \
X0LOOP_CYCLE=cycleNN \
X0LOOP_RUN_STEPS=5000 \
X0LOOP_RESUME=runs/.../ckpt_step_00010000.pt \
X0LOOP_FINAL_FID_ENABLED=false \
X0LOOP_AUX_BATCH_RATIO=0.125 \
X0LOOP_AUX_GRADIENT_RATIO=0.1 \
  experiments/x0loop_v2/run_from_scratch.sh online-x0
```

`X0LOOP_COMPILE_DYNAMIC=true` is an optional, numerically smoke-tested online
rollout optimization. It has a longer initial compilation but avoids static
batch-shape recompilations. It must be recorded in the run configuration and
must not be mixed across compared branches without a matching baseline.

## Evaluation and analysis

Compare aligned Heun trajectories after any training-method change:

```bash
CUDA_VISIBLE_DEVICES=7 uv run python -m \
  experiments.x0loop_v2.analyze_sampling_trajectory \
  --checkpoint fresh=runs/.../fresh/checkpoints/ckpt_step_00015000.pt \
  --checkpoint method=runs/.../method/checkpoints/ckpt_step_00015000.pt \
  --out runs/.../trajectory_analysis_step15000 \
  --seed 20260819 --num-samples 64 --steps 20 --guidance-scale 2.2
```

Report stable step time, throughput, memory, training-core MFU, and method-level
MFU from warm metrics windows:

```bash
uv run python -m experiments.x0loop_v2.analyze_training_efficiency \
  --run fresh=runs/.../fresh \
  --run method=runs/.../method \
  --window-records 100
```

For a candidate that passes its registered 5k-FID gate, evaluate NFE 4/8/20
and then the selected checkpoint with authoritative 50k FID:

```bash
experiments/x0loop_v2/eval_4_8_20.sh CHECKPOINT OUT_DIR
experiments/x0loop_v2/eval_cycle_best_fid50k.sh CYCLE_DIR
```

`run_all_5k.sh`, `run_5k.sh`, and `eval_best_fid50k.sh` are retained for
reproducing the earlier checkpoint-fork falsification experiment. They are not
the current formal from-zero protocol.
