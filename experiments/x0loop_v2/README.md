# x0loop v2 falsification experiment

All four branches resume from the same checkpoint and run exactly 5,000 new
optimizer steps. `FRESH` keeps normal training, `DROP` reproduces the 50%
fresh-data dilution with zero auxiliary loss, `BANK-FIX` uses stratified replay
from an EMA Heun-20/CFG-2.2 trajectory, and `ONLINE` regenerates those states
from the current EMA every step.

The v2 paths keep the full fresh batch. Their auxiliary loss matches the EMA
teacher's CFG velocity on actual inference states; it does not regress to the
ancestral ground truth. The auxiliary coefficient is adjusted using the output
gradient norm, targeting 20% of the fresh output-gradient norm. Metrics include
the achieved ratio, solver index/depth, producer age, and replay size.

Run all branches (GPU 7 by default, GPU 6 fallback):

```bash
experiments/x0loop_v2/run_all_5k.sh path/to/converged_checkpoint.pt
```

Evaluate each resulting checkpoint with the same seed, cyclic label order, EMA,
Heun sampler, and CFG at NFE 4/8/20:

```bash
experiments/x0loop_v2/eval_4_8_20.sh path/to/checkpoint.pt runs/x0loop_v2_eval/fresh
```

The only absolute project configuration path is the shared dataset root,
`/mnt/data/crb/data`. Run outputs and scripts use workspace-relative paths.
