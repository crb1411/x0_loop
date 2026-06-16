#!/usr/bin/env python3
"""Generate v5 sampler-ablation eval experiments.

v5 evaluates time-decayed CFG:
  w(t) = 1 + (w_max - 1) * t^power

`guidance_scale` is w_max. The effective scale decays to 1 as t -> 0.

Output metrics land under:
  /data/seek/aigc/x0_loop/runs/sampler_ablation/v5_616/<exp>/
"""

from __future__ import annotations

import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = (
    "/data/seek/aigc/x0_loop/runs/ablations/cifar10_flow_x0_vloss/jit/"
    "learnable_endpoint/scheme2_beta0p8/20260616_081221/checkpoints/ckpt_step_00100000.pt"
)
OUT_BASE = "/data/seek/aigc/x0_loop/runs/sampler_ablation/v5_616"
NUM_SAMPLES = 10000
BATCH_SIZE = 256

# (sampler, steps, max_cfg, schedule_name, power). schedule_name=None means constant CFG.
GRID = [
    # A) Constant baselines from the best v3/v4 plateau.
    ("heun", 16, 2.25, None, None),
    ("heun", 20, 2.20, None, None),
    ("heun", 20, 2.30, None, None),
    # B) Main dynamic sweep at 20 steps.
    ("heun", 20, 2.60, "power_decay", 1.0),
    ("heun", 20, 3.00, "power_decay", 1.0),
    ("heun", 20, 3.50, "power_decay", 1.0),
    ("heun", 20, 4.00, "power_decay", 1.0),
    ("heun", 20, 2.60, "power_decay", 0.5),
    ("heun", 20, 3.00, "power_decay", 0.5),
    ("heun", 20, 3.50, "power_decay", 0.5),
    ("heun", 20, 3.00, "power_decay", 2.0),
    ("heun", 20, 3.50, "power_decay", 2.0),
    ("heun", 20, 4.00, "power_decay", 2.0),
    # C) Step/cfg probes for the most likely useful dynamic setting range.
    ("heun", 16, 3.00, "power_decay", 1.0),
    ("heun", 16, 3.50, "power_decay", 1.0),
    ("heun", 24, 3.00, "power_decay", 1.0),
    ("heun", 24, 3.50, "power_decay", 1.0),
    ("heun", 16, 3.50, "power_decay", 2.0),
    ("heun", 24, 3.50, "power_decay", 2.0),
    # D) Compare cosine decay against power decay.
    ("heun", 20, 3.00, "cosine_decay", None),
    ("heun", 20, 3.50, "cosine_decay", None),
    ("heun", 16, 3.50, "cosine_decay", None),
    # E) Controls for the v3/v4 secondary sampler.
    ("unipc", 20, 3.00, "power_decay", 1.0),
    ("unipc", 50, 3.00, "power_decay", 1.0),
]


def cfg_tag(cfg: float) -> str:
    return ("%g" % cfg).replace(".", "p")


def power_tag(power: float | None) -> str:
    if power is None:
        return ""
    return "_p" + ("%g" % power).replace(".", "p")


def exp_name(sampler: str, steps: int, cfg: float, schedule: str | None, power: float | None) -> str:
    suffix = "const" if schedule is None else schedule.replace("_decay", "") + power_tag(power)
    return f"{sampler}_s{steps:02d}_maxcfg{cfg_tag(cfg)}_{suffix}"


def schedule_yaml(schedule: str | None, power: float | None) -> str:
    if schedule is None:
        return "  guidance_schedule: null\n"
    lines = [
        "  guidance_schedule:",
        f"    name: {schedule}",
        "    min_scale: 1.0",
    ]
    if power is not None:
        lines.append(f"    power: {power}")
    return "\n".join(lines) + "\n"


EVAL_YAML = """\
# Auto-generated v5_616 sampler-ablation eval override.
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
{schedule_block}  input2: cifar10-train
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
# Auto-generated, fully self-contained single-GPU FID eval for one v5_616 experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "${{ROOT}}"

EXP="{exp}"
CKPT="{ckpt}"
EVAL_CFG="${{ROOT}}/train_run/sampler_ablation/v5_616/${{EXP}}/eval.yaml"
OUT_DIR="{out_base}/${{EXP}}"
LOG_DIR="${{OUT_DIR}}/logs"
mkdir -p "${{LOG_DIR}}"
LOG_FILE="${{LOG_DIR}}/eval.log"

export PYTHONPATH="${{ROOT}}:${{PYTHONPATH:-}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"
PYTHON="${{PYTHON:-/root/miniconda3/envs/vl/bin/python}}"
if [[ ! -x "${{PYTHON}}" ]]; then
  PYTHON="python"
fi

echo "[sampler_ablation_v5_616] exp=${{EXP}} sampler={sampler} steps={steps} max_cfg={cfg} schedule={schedule} power={power} GPU=${{CUDA_VISIBLE_DEVICES}}" | tee -a "${{LOG_FILE}}"
echo "[sampler_ablation_v5_616] out=${{OUT_DIR}}" | tee -a "${{LOG_FILE}}"
echo "[sampler_ablation_v5_616] python=${{PYTHON}}" | tee -a "${{LOG_FILE}}"

"${{PYTHON}}" -m x0loop.eval_fid \\
  --ckpt "${{CKPT}}" \\
  --eval-config "${{EVAL_CFG}}" \\
  --tag "${{EXP}}" \\
  --set "logging.out_dir=${{OUT_DIR}}" 2>&1 | tee -a "${{LOG_FILE}}"

echo "[sampler_ablation_v5_616] done -> ${{OUT_DIR}}/gen_eval_metrics_*.jsonl" | tee -a "${{LOG_FILE}}"
"""


def main() -> None:
    made = []
    for sampler, steps, cfg, schedule, power in GRID:
        name = exp_name(sampler, steps, cfg, schedule, power)
        d = os.path.join(HERE, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "eval.yaml"), "w", encoding="utf-8") as f:
            f.write(EVAL_YAML.format(
                num_samples=NUM_SAMPLES,
                batch_size=BATCH_SIZE,
                steps=steps,
                sampler=sampler,
                cfg=cfg,
                schedule_block=schedule_yaml(schedule, power),
            ))
        run_path = os.path.join(d, "run.sh")
        with open(run_path, "w", encoding="utf-8") as f:
            f.write(RUN_SH.format(
                exp=name,
                ckpt=CKPT,
                out_base=OUT_BASE,
                sampler=sampler,
                steps=steps,
                cfg=cfg,
                schedule=schedule or "constant",
                power="" if power is None else power,
            ))
        os.chmod(run_path, os.stat(run_path).st_mode | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        made.append(name)
    with open(os.path.join(HERE, "experiments.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(made) + "\n")
    print(f"generated {len(made)} experiments under {HERE}:")
    for name in made:
        print("  ", name)


if __name__ == "__main__":
    main()
