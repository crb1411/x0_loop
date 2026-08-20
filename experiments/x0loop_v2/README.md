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

| branch | rollout source | auxiliary target/control |
|---|---|---|
| `fresh` | none | none |
| `fresh-fixed-repro` | none, fixed model time 0.5 | none |
| `fresh-time` | none, explicit path time | none |
| `gated-control` | none, explicit path time | nominal index 16--19 residual, fresh loss only |
| `gated-wide` | none, explicit path time | nominal index 12--19 residual, fresh loss only |
| `gated-dist` | EMA Heun-20 prefix/current suffix | index 16--19 residual + terminal Inception KID |
| `bank-fix` | stratified EMA Heun replay | legacy teacher velocity |
| `bank-x0` | stratified EMA Heun replay | teacher native x0 |
| `online` | current EMA online Heun rollout | legacy teacher velocity |
| `online-x0` | current EMA online Heun rollout | teacher native x0 |
| `online-x0-time` | time-aware EMA online Heun rollout | native x0, 10% parameter-gradient norm |
| `online-x0-time-frozen` | time-aware online Heun rollout from a warmup-frozen EMA | native x0, parameter-gradient norm + cosine |
| `denoise-gan` | fresh noised training state | class-conditional x0 distribution loss |
| `terminal-gan` | EMA Heun-20 prefix, differentiable final Euler | class-conditional terminal distribution loss |

All auxiliary branches retain the complete fresh batch. The target output
gradient ratio, auxiliary batch ratio, compile mode, and resume point must be
explicitly registered in the analysis document before launch. Training and FID
both use the correct learnable endpoint, Heun grid, CFG, EMA, and label rules.

Cycle 03 GAN branches use an auxiliary batch of 32 for a fresh batch of 256
and adapt the generator scale to a measured 10% fresh-output gradient ratio.
`terminal-gan` runs the first 19 Heun intervals under the current EMA with no
gradient and backpropagates only through the sampler's actual final
`t=0.05 -> 0` Euler interval. It rejects any terminal steps/sampler/CFG setting
that differs from `gen_eval`, and cannot be combined with clean-loop replay.

Start a 300-epoch from-zero FRESH run:

```bash
X0LOOP_GPU=7 X0LOOP_CYCLE=cycleNN \
  experiments/x0loop_v2/run_from_scratch.sh fresh
```

Cycle 07 uses a paired seed-43 replication and active-range comparison:

```bash
X0LOOP_GPU=6 X0LOOP_CYCLE=cycle07 X0LOOP_SEED=43 \
  experiments/x0loop_v2/run_from_scratch.sh fresh-time
X0LOOP_GPU=6 X0LOOP_CYCLE=cycle07 X0LOOP_SEED=43 \
  experiments/x0loop_v2/run_from_scratch.sh gated-control
X0LOOP_GPU=6 X0LOOP_CYCLE=cycle07 X0LOOP_SEED=43 \
  experiments/x0loop_v2/run_from_scratch.sh gated-wide
```

For checkpoint-only causal ablation, override
`solver_correction.output_scale=0`; its default is 1. This changes the sampler
and therefore must never be used silently in a formal training evaluation.

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

Cycle 03 shared-prefix distribution screens use the same command shape:

```bash
X0LOOP_GPU=7 \
X0LOOP_CYCLE=cycle03 \
X0LOOP_RUN_STEPS=5000 \
X0LOOP_RESUME=runs/x0loop_v2_from_scratch/cycle01/fresh/checkpoints/ckpt_step_00010000.pt \
X0LOOP_FINAL_FID_ENABLED=false \
  experiments/x0loop_v2/run_from_scratch.sh terminal-gan
```

`X0LOOP_COMPILE_DYNAMIC=true` is an optional, numerically smoke-tested
optimization for the historical output-gradient online branches. It has a
longer initial compilation but avoids static batch-shape recompilations. Exact
parameter-gradient VJPs retain the graph and are incompatible with the current
AOTAutograd donated-buffer dynamic path, so `online-x0-time` formally locks
`compile.dynamic=false`. Compile mode must always be recorded in the run
configuration.

`online-x0-time-frozen` snapshots the EMA exactly once when the configured
warmup boundary is reached. Checkpoints persist that frozen shadow separately
from the moving evaluation EMA, so a post-warmup resume cannot silently change
the teacher. Its metrics include `clean/aux_parameter_cosine` and
`clean/combined_fresh_cosine` in addition to the actual norm ratio. Example
shared-prefix mechanism screen (not a formal from-zero result):

```bash
X0LOOP_GPU=6 X0LOOP_CYCLE=cycle05-screen X0LOOP_RUN_STEPS=5000 \
X0LOOP_RESUME=runs/x0loop_v2_from_scratch/cycle04/online-x0-time/checkpoints/ckpt_step_00010000.pt \
X0LOOP_CLEAN_WARMUP=10000 X0LOOP_AUX_BATCH_RATIO=0.125 \
X0LOOP_AUX_GRADIENT_RATIO=0.10 X0LOOP_FINAL_FID_ENABLED=false \
  experiments/x0loop_v2/run_from_scratch.sh online-x0-time-frozen
```

For an explicitly registered external frozen teacher, also set the
workspace-relative `X0LOOP_TEACHER_CHECKPOINT`. The loader requires an EMA,
matching time-conditioning semantics, and an exact parameter-key match; the
resolved path is stored in the run config. This is an external-teacher
distillation experiment and must not be reported as self-training.

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

Before Cycle 04 generator training, test whether the terminal critic is usable
on a frozen generator and a disjoint held-out set:

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -m experiments.x0loop_v2.diagnose_critic_readiness \
  --checkpoint runs/x0loop_v2_from_scratch/cycle03/terminal-gan/checkpoints/ckpt_step_00015000.pt \
  --out runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000
```

This uses the exact EMA Heun-20/CFG-2.2 kernel, class-matches real and fake
examples, and reports random-init, Cycle 03 co-trained, and freshly trained
critic held-out AUROC. It freezes the generator and therefore is a diagnostic,
not one of the next three x0loop trainings.

Then measure whether the registered 10% output-gradient control remains 10%
after both losses are mapped through the shared backbone:

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -m experiments.x0loop_v2.analyze_gradient_alignment \
  --checkpoint runs/x0loop_v2_from_scratch/cycle03/terminal-gan/checkpoints/ckpt_step_00015000.pt \
  --readiness-critic runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000/fresh_critic_best.pt \
  --fixed-dataset runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000/fixed_terminal_dataset.pt \
  --out runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000
```

Finally, before any DMD generator update, verify that an x0/flow fake score can
adapt to the held-out terminal distribution:

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -m experiments.x0loop_v2.diagnose_fake_score_readiness \
  --checkpoint runs/x0loop_v2_from_scratch/cycle03/terminal-gan/checkpoints/ckpt_step_00015000.pt \
  --fixed-dataset runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000/fixed_terminal_dataset.pt \
  --out runs/x0loop_v2_from_scratch/cycle04/readiness/terminal_step15000/fake_score_x0_v2 \
  --loss-target direct_x0 --time-distribution uniform --min-t 0.05 --max-t 0.95
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
