"""Report stable training throughput and counted MFU for x0loop v2 runs.

The FLOP constant below was measured for the exact CIFAR-10 JiT configuration
with ``torch.utils.flop_counter.FlopCounterMode``.  A batch-256
forward+backward contains 5.154094972928e12 counted FLOPs.  Dividing by three
(one forward plus an approximately two-forward backward) gives a reusable
per-sample forward-equivalent cost for student and EMA model calls.

MFU uses the dense BF16 tensor-core peak of the local 700 W NVIDIA H800
(989 TFLOP/s).  The count excludes optimizer and uncounted elementwise ops, so
it is a reproducible lower-bound-style model FLOP utilization, not GPU busy %.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml


FORWARD_EQUIVALENT_FLOPS_PER_SAMPLE = 6_711_061_162.666667
# Measured with FlopCounterMode for the configured base-channel-16
# discriminator at batch 32: one real/fake D update plus one G-through-D
# backward. Lazy R1 is omitted because nested autograd is not supported by the
# counter; at interval 16 it is a small lower-bound error relative to JiT.
GAN_DISC_FLOPS_BATCH32 = 8_280_670_208.0
DEFAULT_PEAK_TFLOPS = 989.0


@dataclass(frozen=True)
class ComputeEstimate:
    fresh_forward_equivalent_samples: float
    aux_forward_equivalent_samples: float
    teacher_forward_equivalent_samples: float
    extra_flops_per_step: float = 0.0

    @property
    def method_forward_equivalent_samples(self) -> float:
        return (
            self.fresh_forward_equivalent_samples
            + self.aux_forward_equivalent_samples
            + self.teacher_forward_equivalent_samples
        )


def estimate_compute(config: dict) -> ComputeEstimate:
    train = config["train"]
    clean = config.get("clean_loop", {})
    batch = int(train["batch_size"])
    # One trainable forward plus backward is counted as three forwards.
    fresh = 3.0 * batch
    aux = teacher = extra_flops = 0.0
    if bool(clean.get("enabled", False)):
        aux_n = max(1, int(round(batch * float(clean["aux_batch_ratio"]))))
        # CFG concatenates conditional and unconditional batches.  The student
        # auxiliary call is trainable, hence 3 * (2 * aux_n).
        aux = 6.0 * aux_n
        if str(clean.get("aux_gradient_space", "output")) == "parameter":
            # Exact parameter-norm control measures a fresh backward
            # (approximately two forward equivalents per sample) and a CFG
            # auxiliary backward (two equivalents over the doubled batch)
            # before the normal combined backward.
            aux += 2.0 * batch + 4.0 * aux_n
        mode = str(clean["mode"])
        if mode == "online":
            solver_steps = int(clean["solver_steps"])
            # A uniformly selected solver index k uses 2*(k+1) teacher calls,
            # except the final index whose last Euler step uses one fewer call.
            calls_per_root = solver_steps + 1.0 - 1.0 / solver_steps
            teacher = 2.0 * aux_n * calls_per_root  # factor two is CFG batching
        elif mode == "bank_fix":
            refresh_interval = int(clean.get("refresh_interval", 1))
            refresh_n = aux_n * refresh_interval
            roots = max(1, int(round(refresh_n * float(clean["root_fraction"]))))
            advanced = max(0, refresh_n - roots)
            # Root target: one CFG call. Advanced state: two Heun CFG calls plus
            # one accepted-state target CFG call. Amortize over refresh interval.
            teacher = (2.0 * roots + 6.0 * advanced) / refresh_interval

    adversarial = config.get("adversarial", {}) or {}
    if bool(adversarial.get("enabled", False)):
        adv_n = max(1, int(round(batch * float(adversarial.get("batch_ratio", 1.0)))))
        extra_flops += GAN_DISC_FLOPS_BATCH32 * adv_n / 32.0
        if str(adversarial.get("fake_space", "x0_hat")) == "terminal_x0":
            terminal = adversarial.get("terminal", {}) or {}
            steps = int(terminal.get("steps", (config.get("gen_eval", {}) or {}).get("steps", 20)))
            # Each prefix Heun interval has two CFG model calls, each on 2N.
            teacher += 4.0 * adv_n * max(0, steps - 1)
            # The final CFG call is trainable: forward+backward on 2N.
            aux += 6.0 * adv_n
    return ComputeEstimate(fresh, aux, teacher, extra_flops)


def _latest_metrics(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("metrics_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no metrics_*.jsonl under {run_dir}")
    return candidates[-1]


def analyze_run(
    name: str,
    run_dir: Path,
    *,
    window_records: int,
    peak_tflops: float,
) -> dict:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    rows = [json.loads(line) for line in _latest_metrics(run_dir).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "iter_s" in row][-window_records:]
    if not rows:
        raise ValueError(f"no timed metric rows for {run_dir}")

    estimate = estimate_compute(config)
    iter_s = statistics.median(float(row["iter_s"]) for row in rows)
    img_s = statistics.median(float(row["img_s"]) for row in rows)
    fresh_flops = estimate.fresh_forward_equivalent_samples * FORWARD_EQUIVALENT_FLOPS_PER_SAMPLE
    method_flops = (
        estimate.method_forward_equivalent_samples * FORWARD_EQUIVALENT_FLOPS_PER_SAMPLE
        + estimate.extra_flops_per_step
    )
    peak_flops_s = peak_tflops * 1.0e12
    return {
        "name": name,
        "records": len(rows),
        "step_first": int(rows[0]["step"]),
        "step_last": int(rows[-1]["step"]),
        "iter_s_median": iter_s,
        "img_s_median": img_s,
        "gpu_mem_gb_max": max(float(row.get("gpu_mem_gb", 0.0)) for row in rows),
        "fresh_tflop_per_step": fresh_flops / 1.0e12,
        "method_tflop_per_step": method_flops / 1.0e12,
        "training_core_mfu_pct": 100.0 * fresh_flops / iter_s / peak_flops_s,
        "method_mfu_pct": 100.0 * method_flops / iter_s / peak_flops_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--window-records", type=int, default=100)
    parser.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    args = parser.parse_args()
    if args.window_records <= 0:
        raise ValueError("--window-records must be positive")

    reports = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"--run must be NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        reports.append(
            analyze_run(
                name,
                Path(directory),
                window_records=args.window_records,
                peak_tflops=args.peak_tflops,
            )
        )

    print("| run | steps | median s/step | img/s | peak GiB | core MFU | method MFU | method TF/step |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for report in reports:
        print(
            f"| {report['name']} | {report['step_first']}-{report['step_last']} | "
            f"{report['iter_s_median']:.4f} | {report['img_s_median']:.1f} | "
            f"{report['gpu_mem_gb_max']:.2f} | {report['training_core_mfu_pct']:.2f}% | "
            f"{report['method_mfu_pct']:.2f}% | {report['method_tflop_per_step']:.3f} |"
        )


if __name__ == "__main__":
    main()
