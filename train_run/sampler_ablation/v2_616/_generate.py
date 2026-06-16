#!/usr/bin/env python3
"""Generate v2 sampler-ablation eval experiments.

v2 focuses on the v1 best region:
  - dpmpp_2m around cfg=2.0
  - dpmpp_2m step sweep around 20
  - small heun/unipc controls at stronger CFG

Output metrics land under:
  /data/seek/aigc/x0_loop/runs/sampler_ablation/v2_616/<exp>/
"""

from __future__ import annotations

import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = (
    "/data/seek/aigc/x0_loop/runs/ablations/cifar10_flow_x0_vloss/jit/"
    "learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
)
OUT_BASE = "/data/seek/aigc/x0_loop/runs/sampler_ablation/v2_616"
NUM_SAMPLES = 10000
BATCH_SIZE = 256

# (sampler, steps, cfg). Each row is one independent single-GPU experiment.
GRID = [
    # A) Fine CFG sweep around the v1 winner: dpmpp_2m_s20_cfg2.
    ("dpmpp_2m", 20, 1.75),
    ("dpmpp_2m", 20, 2.00),
    ("dpmpp_2m", 20, 2.25),
    ("dpmpp_2m", 20, 2.50),
    ("dpmpp_2m", 20, 2.75),
    # B) Step sweep at cfg=2.0.
    ("dpmpp_2m", 12, 2.00),
    ("dpmpp_2m", 16, 2.00),
    ("dpmpp_2m", 24, 2.00),
    ("dpmpp_2m", 30, 2.00),
    ("dpmpp_2m", 40, 2.00),
    # C) Coupled step/cfg probes near the likely optimum.
    ("dpmpp_2m", 16, 2.25),
    ("dpmpp_2m", 24, 2.25),
    ("dpmpp_2m", 30, 2.25),
    # D) Controls: see whether heun/unipc also benefit from stronger CFG.
    ("heun", 20, 2.00),
    ("heun", 30, 2.00),
    ("heun", 50, 2.00),
    ("unipc", 20, 2.00),
    ("unipc", 50, 2.00),
]


def cfg_tag(cfg: float) -> str:
    return ("%g" % cfg).replace(".", "p")


def exp_name(sampler: str, steps: int, cfg: float) -> str:
    return f"{sampler}_s{steps:02d}_cfg{cfg_tag(cfg)}"


EVAL_YAML = """\
# Auto-generated v2_616 sampler-ablation eval override.
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
  datasets_root: /root/data/cifar10_data
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
# Auto-generated, fully self-contained single-GPU FID eval for one v2_616 experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${{ROOT}}"

EXP="{exp}"
CKPT="{ckpt}"
EVAL_CFG="${{ROOT}}/train_run/sampler_ablation/v2_616/${{EXP}}/eval.yaml"
OUT_DIR="{out_base}/${{EXP}}"
LOG_DIR="${{OUT_DIR}}/logs"
mkdir -p "${{LOG_DIR}}"
LOG_FILE="${{LOG_DIR}}/eval.log"

export PYTHONPATH="${{ROOT}}:${{PYTHONPATH:-}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"

echo "[sampler_ablation_v2_616] exp=${{EXP}} sampler={sampler} steps={steps} cfg={cfg} GPU=${{CUDA_VISIBLE_DEVICES}}" | tee -a "${{LOG_FILE}}"
echo "[sampler_ablation_v2_616] out=${{OUT_DIR}}" | tee -a "${{LOG_FILE}}"

python -m x0loop.eval_fid \\
  --ckpt "${{CKPT}}" \\
  --eval-config "${{EVAL_CFG}}" \\
  --tag "${{EXP}}" \\
  --set "logging.out_dir=${{OUT_DIR}}" 2>&1 | tee -a "${{LOG_FILE}}"

echo "[sampler_ablation_v2_616] done -> ${{OUT_DIR}}/gen_eval_metrics_*.jsonl" | tee -a "${{LOG_FILE}}"
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
    print(f"generated {len(made)} experiments under {HERE}:")
    for name in made:
        print("  ", name)


if __name__ == "__main__":
    main()
