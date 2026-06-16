#!/usr/bin/env python3
"""Generate self-contained sampler/steps/cfg FID-eval experiments.

Each experiment gets its own folder under train_run/sampler_ablation/v1/<exp>/ with:
  - eval.yaml : override config (sampler, steps, guidance_scale, num_samples, ...)
  - run.sh    : fully independent one-click launcher (no shared sourcing)

Re-run this script to regenerate after editing the GRID below.
Output metrics land under: /data/seek/aigc/x0_loop/runs/sampler_ablation/v1/<exp>/
"""

from __future__ import annotations

import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = (
    "/data/seek/aigc/x0_loop/runs/ablations/cifar10_flow_x0_vloss/jit/"
    "learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
)
OUT_BASE = "/data/seek/aigc/x0_loop/runs/sampler_ablation/v1"
NUM_SAMPLES = 10000
BATCH_SIZE = 256

# (sampler, steps, cfg). Each row is one independent experiment.
GRID = [
    # A) sampler comparison @ steps=20, cfg=1.5
    ("euler", 20, 1.5),
    ("heun", 20, 1.5),
    ("dpmpp_2m", 20, 1.5),
    ("unipc", 20, 1.5),
    ("unipc3", 20, 1.5),
    # B) step sweep @ unipc, cfg=1.5
    ("unipc", 6, 1.5),
    ("unipc", 10, 1.5),
    ("unipc", 16, 1.5),
    ("unipc", 50, 1.5),
    # C) cfg sweep @ dpmpp_2m, steps=20
    ("dpmpp_2m", 20, 1.0),
    ("dpmpp_2m", 20, 2.0),
    ("dpmpp_2m", 20, 3.0),
    # reference: high-quality heun
    ("heun", 50, 1.5),
]


def cfg_tag(cfg: float) -> str:
    return ("%g" % cfg).replace(".", "p")


def exp_name(sampler: str, steps: int, cfg: float) -> str:
    return f"{sampler}_s{steps:02d}_cfg{cfg_tag(cfg)}"


EVAL_YAML = """\
# Auto-generated sampler-ablation eval override (merged over the checkpoint's config).
# Edit + re-run ./run.sh for a one-click single-GPU FID/IS/KID eval.
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
# Auto-generated, fully self-contained single-GPU FID eval for one (sampler, steps, cfg).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${{ROOT}}"

EXP="{exp}"
CKPT="{ckpt}"
EVAL_CFG="${{ROOT}}/train_run/sampler_ablation/v1/${{EXP}}/eval.yaml"
OUT_DIR="{out_base}/${{EXP}}"
LOG_DIR="${{OUT_DIR}}/logs"
mkdir -p "${{LOG_DIR}}"
LOG_FILE="${{LOG_DIR}}/eval.log"

export PYTHONPATH="${{ROOT}}:${{PYTHONPATH:-}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"

echo "[sampler_ablation] exp=${{EXP}} sampler={sampler} steps={steps} cfg={cfg} GPU=${{CUDA_VISIBLE_DEVICES}}" | tee -a "${{LOG_FILE}}"
echo "[sampler_ablation] out=${{OUT_DIR}}" | tee -a "${{LOG_FILE}}"

python -m x0loop.eval_fid \\
  --ckpt "${{CKPT}}" \\
  --eval-config "${{EVAL_CFG}}" \\
  --tag "${{EXP}}" \\
  --set "logging.out_dir=${{OUT_DIR}}" 2>&1 | tee -a "${{LOG_FILE}}"

echo "[sampler_ablation] done -> ${{OUT_DIR}}/gen_eval_metrics_*.jsonl" | tee -a "${{LOG_FILE}}"
"""


def main():
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
    for n in made:
        print("  ", n)


if __name__ == "__main__":
    main()
