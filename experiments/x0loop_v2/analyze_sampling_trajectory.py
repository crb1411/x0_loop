"""Compare fixed-root Heun trajectories from multiple x0loop checkpoints.

Example:
    CUDA_VISIBLE_DEVICES=7 uv run python -m experiments.x0loop_v2.analyze_sampling_trajectory \
      --checkpoint fresh=runs/.../fresh/checkpoints/ckpt_step_00058500.pt \
      --checkpoint bank_fix=runs/.../bank-fix/checkpoints/ckpt_step_00058500.pt \
      --checkpoint online=runs/.../online/checkpoints/ckpt_step_00058500.pt \
      --out runs/x0loop_v2_from_scratch/cycle01/trajectory_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from typing import Any

import torch

from x0loop.infer import _load_model
from x0loop.training.sampling import build_null_class_cond, build_sample_cond, save_sample_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare aligned Heun sampling trajectories.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named checkpoint; pass at least two times.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.2)
    return parser.parse_args()


def _named_checkpoints(values: list[str]) -> list[tuple[str, str]]:
    checkpoints: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"Checkpoint names must be non-empty and unique, got {name!r}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        checkpoints.append((name, path))
        seen.add(name)
    if len(checkpoints) < 2:
        raise ValueError("Pass at least two --checkpoint NAME=PATH arguments.")
    return checkpoints


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def _step_rows(trace: list[dict[str, Any]], final: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    previous_x0: torch.Tensor | None = None
    for index, item in enumerate(trace):
        x = item["x"].float()
        x_next = item["x_next"].float()
        x0_hat = item["x0_hat"].float()
        accepted_delta = x_next - x
        row = {
            "solver_index": index,
            "t": float(item["t"]),
            "s": float(item["s"]),
            "state_rms": _rms(x),
            "accepted_delta_rms": _rms(accepted_delta),
            "x0_hat_rms": _rms(x0_hat),
            "x0_drift_rms": 0.0 if previous_x0 is None else _rms(x0_hat - previous_x0),
            "x0_to_final_rms": _rms(x0_hat - final),
        }
        if "velocity" in item:
            row["velocity_rms"] = _rms(item["velocity"])
        if "x_euler" in item:
            correction = x_next - item["x_euler"].float()
            row["heun_correction_rms"] = _rms(correction)
            row["relative_heun_correction"] = _rms(correction) / max(_rms(accepted_delta), 1e-12)
        rows.append(row)
        previous_x0 = x0_hat
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_markdown(summary: dict[str, Any], path: str) -> None:
    lines = [
        "# Fixed-root Heun trajectory comparison",
        "",
        f"seed={summary['seed']}, samples={summary['num_samples']}, "
        f"steps={summary['steps']}, CFG={summary['guidance_scale']}",
        "",
        "## Per-model summary",
        "",
        "| model | endpoint RMS | final RMS | mean Heun correction | max relative correction | final x0 gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in summary["models"].items():
        rows = model["steps"]
        corrections = [row.get("heun_correction_rms", 0.0) for row in rows]
        relative = [row.get("relative_heun_correction", 0.0) for row in rows]
        lines.append(
            f"| {name} | {rows[0]['state_rms']:.6f} | {model['final_rms']:.6f} | "
            f"{sum(corrections) / len(corrections):.6f} | {max(relative):.6f} | "
            f"{rows[-1]['x0_to_final_rms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise final-sample distance",
            "",
            "| pair | RMS distance |",
            "|---|---:|",
        ]
    )
    for pair, distance in summary["pairwise_final_rms"].items():
        lines.append(f"| {pair} | {distance:.6f} |")
    lines.extend(
        [
            "",
            "The JSON file contains every solver-step statistic. Raw traces and final grids are saved beside this report.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = _parse_args()
    checkpoints = _named_checkpoints(args.checkpoint)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    summary: dict[str, Any] = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "sampler": "heun",
        "guidance_scale": args.guidance_scale,
        "models": {},
        "pairwise_final_rms": {},
    }
    finals: dict[str, torch.Tensor] = {}

    for name, checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = dict(checkpoint.get("config") or {})
        if not cfg:
            raise ValueError(f"Checkpoint {checkpoint_path} has no embedded config.")
        model, model_cfg, process = _load_model(cfg, checkpoint_path, device)
        model.eval()
        cond = build_sample_cond(cfg, sample_num=args.num_samples, device=device)
        null_cond = build_null_class_cond(cfg, sample_num=args.num_samples, device=device)

        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed)
            result = process.sample(
                model=model,
                steps=args.steps,
                shape=(args.num_samples, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
                device=device,
                dtype=torch.float32,
                return_trace=True,
                cond=cond,
                null_cond=null_cond,
                guidance_scale=args.guidance_scale,
                sampler="heun",
            )

        final = result["x"].detach().cpu().float()
        trace = result["trace"]
        finals[name] = final
        summary["models"][name] = {
            "checkpoint": checkpoint_path,
            "checkpoint_step": int(checkpoint.get("step", 0)),
            "final_rms": _rms(final),
            "final_mean": float(final.mean()),
            "final_std": float(final.std(unbiased=False)),
            "steps": _step_rows(trace, final),
        }
        torch.save(trace, os.path.join(args.out, f"{name}_trace.pt"))
        save_sample_grid(final, os.path.join(args.out, f"{name}_final_grid.png"), cfg=cfg)
        del model, process, checkpoint, result, trace
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for left, right in combinations(finals, 2):
        summary["pairwise_final_rms"][f"{left} vs {right}"] = _rms(finals[left] - finals[right])

    json_path = os.path.join(args.out, "trajectory_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown_path = os.path.join(args.out, "trajectory_summary.md")
    _write_markdown(summary, markdown_path)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
