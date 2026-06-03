#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(
    r"\[(?P<lo>[0-9.]+),(?P<hi>[0-9.]+)[)\]]: "
    r"n=(?P<n>[0-9]+), "
    r"a=(?P<a>[-+0-9.eE]+), "
    r"w=(?P<w>[-+0-9.eE]+), "
    r"leps=(?P<leps>[-+0-9.eE]+), "
    r"lx0=(?P<lx0>[-+0-9.eE]+), "
    r"lv=(?P<lv>[-+0-9.eE]+)"
)
EPS = 1e-12


@dataclass
class RunMetrics:
    path: Path
    label: str
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def load_run(path: Path, label: str | None = None) -> RunMetrics:
    rows = read_jsonl(path)
    train = [r for r in rows if "loss" in r and "step" in r]
    eval_rows = [r for r in rows if "step" in r and any(k.startswith("eval/") for k in r)]
    if not train:
        raise ValueError(f"No training rows with key 'loss' found in {path}")
    return RunMetrics(path=path, label=label or path.parent.name, train=train, eval=eval_rows)


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            xs.append(int(row["step"]))
            ys.append(float(value))
    return xs, ys


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    total = 0.0
    q: list[float] = []
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.pop(0)
        out.append(total / len(q))
    return out


def positive(values: list[float]) -> list[float]:
    return [max(float(v), EPS) for v in values]


def robust_ylim(ax, values: list[float], *, log: bool) -> None:
    if not values:
        return
    import numpy as np

    arr = np.asarray([v for v in values if math.isfinite(v) and (v > 0 if log else True)], dtype=float)
    if arr.size < 4:
        return
    lo, hi = np.percentile(arr, [1.0, 99.5])
    if log:
        lo = max(lo, EPS)
    if hi > lo:
        pad = 0.15 if log else 0.05
        ax.set_ylim(lo * (1.0 - pad) if not log else lo / 1.5, hi * (1.0 + pad) if not log else hi * 1.5)


def plot_raw_and_smooth(ax, xs: list[int], ys: list[float], *, label: str, smooth: int, log: bool = False, **kwargs) -> list[float]:
    if not xs:
        return []
    raw = positive(ys) if log else ys
    smooth_y = moving_average(raw, smooth)
    color = kwargs.pop("color", None)
    ax.plot(xs, raw, color=color, alpha=0.18, linewidth=0.6)
    line = ax.plot(xs, smooth_y, label=label, color=color, linewidth=1.6, **kwargs)
    return smooth_y if line else []


def parse_summary(summary: str) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    for match in SUMMARY_RE.finditer(summary):
        bins.append({k: float(v) for k, v in match.groupdict().items()})
    return bins


def row_summary(row: dict[str, Any]) -> str | None:
    for key in ("summary", "eval/summary"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def summary_rows(rows: list[dict[str, Any]], max_rows: int = 0) -> list[tuple[int, list[dict[str, float]]]]:
    parsed: list[tuple[int, list[dict[str, float]]]] = []
    for row in rows:
        text = row_summary(row)
        if text is None:
            continue
        bins = parse_summary(text)
        if bins:
            parsed.append((int(row["step"]), bins))
    if max_rows > 0 and len(parsed) > max_rows:
        stride = math.ceil(len(parsed) / max_rows)
        parsed = parsed[::stride]
    return parsed


def final_bins(rows: list[dict[str, Any]]) -> tuple[int | None, list[dict[str, float]]]:
    for row in reversed(rows):
        text = row_summary(row)
        if text is None:
            continue
        bins = parse_summary(text)
        if bins:
            return int(row["step"]), bins
    return None, []


def output_paths(metrics_path: Path, output: Path | None) -> tuple[Path, Path]:
    if output is not None:
        if output.suffix.lower() == ".png":
            return output, output.with_name(output.stem + "_timebins.png")
        output.mkdir(parents=True, exist_ok=True)
        base = metrics_path.stem
        return output / f"{base}_dashboard.png", output / f"{base}_timebins.png"
    return (
        metrics_path.with_name(f"{metrics_path.stem}_dashboard.png"),
        metrics_path.with_name(f"{metrics_path.stem}_timebins.png"),
    )


def plot_dashboard(run: RunMetrics, out_path: Path, *, smooth: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    ax_obj, ax_diag, ax_eval, ax_grad, ax_speed, ax_ratio = axes.ravel()
    fig.suptitle(f"{run.label} | {run.path.name} | steps {run.train[0]['step']}..{run.train[-1]['step']}", fontsize=13)

    values_for_ylim: list[float] = []
    xs, ys = series(run.train, "loss")
    values_for_ylim += plot_raw_and_smooth(ax_obj, xs, ys, label="objective loss (may be weighted)", smooth=smooth, log=True)
    ax_obj.set_title("Objective Loss")
    ax_obj.set_yscale("log")
    ax_obj.set_xlabel("step")
    ax_obj.set_ylabel("log scale")
    ax_obj.grid(True, alpha=0.25)
    ax_obj.legend(fontsize=8)
    robust_ylim(ax_obj, values_for_ylim, log=True)

    values_for_ylim = []
    for key, label in (("loss_x0", "x0"), ("loss_v", "v"), ("loss_eps", "eps")):
        xs, ys = series(run.train, key)
        values_for_ylim += plot_raw_and_smooth(ax_diag, xs, ys, label=label, smooth=smooth, log=True)
    ax_diag.set_title("Train Diagnostics, Unweighted")
    ax_diag.set_yscale("log")
    ax_diag.set_xlabel("step")
    ax_diag.grid(True, alpha=0.25)
    ax_diag.legend(fontsize=8)
    robust_ylim(ax_diag, values_for_ylim, log=True)

    values_for_ylim = []
    for key, label in (
        ("eval/loss_x0", "eval x0"),
        ("eval/loss_v", "eval v"),
        ("eval/loss_no_weight", "eval no weight"),
        ("eval/loss_weighted", "eval weighted"),
    ):
        xs, ys = series(run.eval, key)
        if xs:
            vals = positive(ys)
            values_for_ylim += vals
            ax_eval.plot(xs, vals, marker="o", markersize=3, linewidth=1.2, label=label)
    ax_eval.set_title("Eval Metrics")
    ax_eval.set_yscale("log")
    ax_eval.set_xlabel("step")
    ax_eval.grid(True, alpha=0.25)
    ax_eval.legend(fontsize=8)
    robust_ylim(ax_eval, values_for_ylim, log=True)

    xs, grad = series(run.train, "grad_norm")
    if xs:
        ax_grad.plot(xs, positive(moving_average(grad, smooth)), label="grad_norm", color="tab:red")
    _, clips = series(run.train, "grad_clip")
    if clips:
        ax_grad.axhline(clips[-1], color="gray", linestyle="--", linewidth=1.0, label=f"clip={clips[-1]:g}")
    ax_grad.set_title("Gradient Norm / Clip")
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("step")
    ax_grad.grid(True, alpha=0.25)
    ax_grad.legend(fontsize=8)

    xs, img_s = series(run.train, "img_s")
    if xs:
        ax_speed.plot(xs, moving_average(img_s, smooth), label="img/s", color="tab:orange")
    ax_speed_2 = ax_speed.twinx()
    xs, lr = series(run.train, "lr")
    if xs:
        ax_speed_2.plot(xs, lr, label="lr", color="tab:blue", linestyle="--")
    ax_speed.set_title("Throughput and LR")
    ax_speed.set_xlabel("step")
    ax_speed.set_ylabel("img/s")
    ax_speed_2.set_ylabel("lr")
    ax_speed.grid(True, alpha=0.25)

    xs_x0, x0 = series(run.train, "loss_x0")
    xs_v, v = series(run.train, "loss_v")
    if xs_x0 and xs_v and xs_x0 == xs_v:
        ratio = [max(vv, EPS) / max(xx, EPS) for vv, xx in zip(v, x0)]
        ax_ratio.plot(xs_x0, positive(moving_average(ratio, smooth)), label="loss_v / loss_x0", color="tab:purple")
    xs_eps, eps = series(run.train, "loss_eps")
    if xs_x0 and xs_eps and xs_x0 == xs_eps:
        ratio = [max(ee, EPS) / max(xx, EPS) for ee, xx in zip(eps, x0)]
        ax_ratio.plot(xs_x0, positive(moving_average(ratio, smooth)), label="loss_eps / loss_x0", color="tab:brown")
    ax_ratio.set_title("Diagnostic Ratios")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("step")
    ax_ratio.grid(True, alpha=0.25)
    ax_ratio.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def heatmap_data(parsed: list[tuple[int, list[dict[str, float]]]], key: str):
    import numpy as np

    steps = [step for step, _ in parsed]
    centers = [(b["lo"] + b["hi"]) * 0.5 for b in parsed[-1][1]]
    data = np.asarray([[b[key] for b in bins] for _, bins in parsed], dtype=float).T
    return steps, centers, data


def draw_heatmap(fig, ax, parsed: list[tuple[int, list[dict[str, float]]]], key: str, title: str, *, log_color: bool) -> None:
    import numpy as np

    if not parsed:
        ax.set_title(f"{title} unavailable")
        ax.axis("off")
        return
    steps, centers, data = heatmap_data(parsed, key)
    label = key
    if log_color:
        data = np.log10(np.clip(data, EPS, None))
        label = f"log10 {key}"
    finite = data[np.isfinite(data)]
    kwargs = {}
    if finite.size > 4:
        kwargs["vmin"], kwargs["vmax"] = np.percentile(finite, [1.0, 99.0])
    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[steps[0], steps[-1], centers[0], centers[-1]],
        interpolation="nearest",
        **kwargs,
    )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("t-bin center")
    fig.colorbar(image, ax=ax, label=label)


def plot_timebins(run: RunMetrics, out_path: Path, *, max_heatmap_rows: int) -> None:
    import matplotlib.pyplot as plt

    parsed_train = summary_rows(run.train, max_rows=max_heatmap_rows)
    final_step, bins = final_bins(run.eval or run.train)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_lx0 = fig.add_subplot(gs[0, 0])
    ax_lv = fig.add_subplot(gs[0, 1])
    ax_leps = fig.add_subplot(gs[0, 2])
    ax_profile = fig.add_subplot(gs[1, :2])
    ax_count = fig.add_subplot(gs[1, 2])
    fig.suptitle(f"{run.label} | t-bin diagnostics | {run.path.name}", fontsize=13)

    draw_heatmap(fig, ax_lx0, parsed_train, "lx0", "t-bin x0 MSE", log_color=True)
    draw_heatmap(fig, ax_lv, parsed_train, "lv", "t-bin v MSE", log_color=True)
    draw_heatmap(fig, ax_leps, parsed_train, "leps", "t-bin eps MSE", log_color=True)

    if bins:
        t = [(b["lo"] + b["hi"]) * 0.5 for b in bins]
        metric_lines = [("lx0", "x0 MSE"), ("lv", "v MSE"), ("leps", "eps MSE")]
        for key, label in metric_lines:
            ax_profile.plot(t, positive([b[key] for b in bins]), marker="o", label=label)
        ax_profile.set_yscale("log")
        ax_profile.set_xlabel("t-bin center")
        ax_profile.set_title(f"Final Profile at step {final_step}")
        ax_profile.grid(True, alpha=0.25)
        ax_profile.legend(fontsize=8)
        ax_w = ax_profile.twinx()
        ax_w.plot(t, [b["w"] for b in bins], color="tab:red", linestyle="--", marker="x", label="weight")
        ax_w.set_ylabel("weight")
        ax_w.legend(fontsize=8, loc="lower right")

        ax_count.bar(t, [b["n"] for b in bins], width=0.04, color="tab:gray")
        ax_count.set_title("Final t-bin Counts")
        ax_count.set_xlabel("t-bin center")
        ax_count.set_ylabel("n")
        ax_count.grid(True, alpha=0.25)
    else:
        ax_profile.set_title("Final profile unavailable")
        ax_profile.axis("off")
        ax_count.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_comparison(runs: list[RunMetrics], out_path: Path, *, smooth: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], "eval/loss_x0", "Eval x0 MSE", True, True),
        (axes[0, 1], "eval/loss_v", "Eval v MSE", True, True),
        (axes[0, 2], "eval/loss_no_weight", "Eval No-weight Objective", True, True),
        (axes[1, 0], "loss_x0", "Train x0 Diagnostic", True, False),
        (axes[1, 1], "loss_v", "Train v Diagnostic", True, False),
        (axes[1, 2], "grad_norm", "Gradient Norm", True, False),
    ]
    fig.suptitle("Comparable metrics only. Raw objective loss is intentionally excluded.", fontsize=13)
    for ax, key, title, log, is_eval in panels:
        for run in runs:
            rows = run.eval if is_eval else run.train
            xs, ys = series(rows, key)
            if not xs:
                continue
            vals = positive(ys) if log else ys
            if not is_eval:
                vals = moving_average(vals, smooth)
            ax.plot(xs, vals, marker="o" if is_eval else None, markersize=3, linewidth=1.4, label=run.label)
        ax.set_title(title)
        ax.set_xlabel("step")
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot x0loop training trends from one or more metrics_*.jsonl files.")
    parser.add_argument("metrics_jsonl", nargs="+", type=Path, help="Path(s) to metrics_*.jsonl")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path or directory")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for multiple input files")
    parser.add_argument("--smooth", type=int, default=20, help="Moving-average window for train curves")
    parser.add_argument("--max-heatmap-rows", type=int, default=800, help="Downsample summary heatmap rows to this count")
    args = parser.parse_args()

    labels = args.labels or []
    if labels and len(labels) != len(args.metrics_jsonl):
        raise ValueError("--labels length must match number of metrics files")
    runs = [load_run(path, labels[i] if labels else None) for i, path in enumerate(args.metrics_jsonl)]

    if len(runs) == 1:
        dashboard_path, timebin_path = output_paths(runs[0].path, args.output)
        plot_dashboard(runs[0], dashboard_path, smooth=max(1, args.smooth))
        plot_timebins(runs[0], timebin_path, max_heatmap_rows=max(0, args.max_heatmap_rows))
        print(dashboard_path)
        print(timebin_path)
    else:
        if args.output is None:
            out_path = Path("comparison_metrics.png")
        elif args.output.suffix.lower() == ".png":
            out_path = args.output
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            out_path = args.output / "comparison_metrics.png"
        plot_comparison(runs, out_path, smooth=max(1, args.smooth))
        print(out_path)


if __name__ == "__main__":
    main()
