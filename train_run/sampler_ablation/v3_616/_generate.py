#!/usr/bin/env python3
"""Generate v3 sampler-ablation eval experiments.

v3 broadens the CFG range after v2 showed heun/cfg=2.0 is strongest:
  - broad heun CFG sweep at 20 steps
  - heun step sweep at cfg=2.0
  - coupled heun probes around cfg=1.9/2.1
  - small unipc/dpmpp controls at wider CFG

Output metrics land under:
  ./runs/sampler_ablation/v3_616/<exp>/
"""

from __future__ import annotations

import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = (
    "./runs/ablations/cifar10_flow_x0_vloss/jit/"
    "learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
)
OUT_BASE = "./runs/sampler_ablation/v3_616"
NUM_SAMPLES = 10000
BATCH_SIZE = 256

# (sampler, steps, cfg). Each row is one independent single-GPU experiment.
GRID = [
    # A) Wide CFG sweep for the current best sampler/step setting.
    ("heun", 20, 1.00),
    ("heun", 20, 1.40),
    ("heun", 20, 1.70),
    ("heun", 20, 1.90),
    ("heun", 20, 2.00),
    ("heun", 20, 2.10),
    ("heun", 20, 2.30),
    ("heun", 20, 2.60),
    ("heun", 20, 3.00),
    ("heun", 20, 3.50),
    # B) Step sweep at cfg=2.0, including low-step speed points.
    ("heun", 8, 2.00),
    ("heun", 12, 2.00),
    ("heun", 16, 2.00),
    ("heun", 24, 2.00),
    ("heun", 30, 2.00),
    ("heun", 40, 2.00),
    ("heun", 50, 2.00),
    # C) Coupled probes around the likely local optimum.
    ("heun", 16, 1.90),
    ("heun", 16, 2.10),
    ("heun", 24, 1.90),
    ("heun", 24, 2.10),
    ("heun", 30, 1.90),
    ("heun", 30, 2.10),
    # D) Controls: test whether other samplers benefit from wider CFG.
    ("unipc", 20, 1.70),
    ("unipc", 20, 2.30),
    ("unipc", 50, 2.30),
    ("dpmpp_2m", 30, 1.75),
    ("dpmpp_2m", 30, 2.50),
    ("dpmpp_2m", 30, 3.00),
]


def cfg_tag(cfg: float) -> str:
    return ("%g" % cfg).replace(".", "p")


def exp_name(sampler: str, steps: int, cfg: float) -> str:
    return f"{sampler}_s{steps:02d}_cfg{cfg_tag(cfg)}"


EVAL_YAML = """\
# Auto-generated v3_616 sampler-ablation eval override.
distributed:
  mode: none
compile:
  enabled: false
eval:
  enabled: false
logging:
  use_tb: false
gen_eval:
  enabled: true
  num_samples: {num_samples}
  batch_size: {batch_size}
  steps: {steps}
  sampler: {sampler}
  guidance_scale: {cfg}
  input2: cifar10-train
  datasets_root: /mnt/data/crb/data
  datasets_download: false
  keep_images: false
  keep_images_count: 100
  metrics:
    isc: true
    fid: true
    kid: true
    prc: true
    mind: false
    ppl: false
  final:
    enabled: false
"""

RUN_SH = """\
#!/usr/bin/env bash
# Auto-generated, fully self-contained single-GPU FID eval for one v3_616 experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${{ROOT}}"

EXP="{exp}"
CKPT="{ckpt}"
EVAL_CFG="${{ROOT}}/train_run/sampler_ablation/v3_616/${{EXP}}/eval.yaml"
OUT_DIR="{out_base}/${{EXP}}"
LOG_DIR="${{OUT_DIR}}/logs"
mkdir -p "${{LOG_DIR}}"
LOG_FILE="${{LOG_DIR}}/eval.log"

export PYTHONPATH="${{ROOT}}:${{PYTHONPATH:-}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"

echo "[sampler_ablation_v3_616] exp=${{EXP}} sampler={sampler} steps={steps} cfg={cfg} GPU=${{CUDA_VISIBLE_DEVICES}}" | tee -a "${{LOG_FILE}}"
echo "[sampler_ablation_v3_616] out=${{OUT_DIR}}" | tee -a "${{LOG_FILE}}"

uv run python -m x0loop.eval_fid \\
  --ckpt "${{CKPT}}" \\
  --eval-config "${{EVAL_CFG}}" \\
  --tag "${{EXP}}" \\
  --set "logging.out_dir=${{OUT_DIR}}" 2>&1 | tee -a "${{LOG_FILE}}"

echo "[sampler_ablation_v3_616] done -> ${{OUT_DIR}}/gen_eval_metrics_*.jsonl" | tee -a "${{LOG_FILE}}"
"""


def main() -> None:
    made = []
    for sampler, steps, cfg in GRID:
        name = exp_name(sampler, steps, cfg)
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "eval.yaml"), "w", encoding="utf-8") as f:
            f.write(EVAL_YAML.format(num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE,
                                     steps=steps, sampler=sampler, cfg=cfg))
        run_path = os.path.join(d, "run.sh")
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(RUN_SH.format(exp=name, ckpt=CKPT, out_base=OUT_BASE,
                                  sampler=sampler, steps=steps, cfg=cfg))
        os.chmod(run_path, os.stat(run_path).st_mode | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        made.append(name)
    with open(os.path.join(HERE, "experiments.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(made) + "\n")
    print(f"generated {len(made)} experiments under {HERE}:")
    for name in made:
        print("  ", name)


if __name__ == "__main__":
    main()
