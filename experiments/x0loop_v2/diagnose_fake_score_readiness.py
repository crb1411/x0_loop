"""Train only the DMD fake score on a fixed terminal distribution.

The generator and real-score teacher are frozen.  This is the second Cycle 04
readiness gate: a fake score may guide the generator only if it estimates the
held-out generated distribution better than the unadapted real teacher.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from x0loop.core.time_sampling import build_time_sampler
from x0loop.losses.spec import build_loss
from x0loop.models.denoiser import Denoiser
from x0loop.models.factory import build_model
from x0loop.training.factories import build_process, build_schedule
from x0loop.utils.checkpoint import _adapt_ema_state_to_model, _load_model_state_with_fallback
from x0loop.utils.ema import EMA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DMD fake-score readiness on fixed terminal samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed-dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss-target", choices=("native_v", "direct_x0"), default="native_v")
    parser.add_argument("--time-distribution", choices=("native", "uniform"), default="native")
    parser.add_argument("--min-t", type=float, default=0.001)
    parser.add_argument("--max-t", type=float, default=0.999)
    parser.add_argument("--improvement-threshold", type=float, default=0.10)
    parser.add_argument("--max-bin-regression", type=float, default=0.10)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def readiness_decision(
    teacher: dict[str, Any],
    fake: dict[str, Any],
    *,
    improvement_threshold: float,
    max_bin_regression: float,
) -> dict[str, Any]:
    ratios = {
        key: float(fake[key]) / max(float(teacher[key]), 1e-12)
        for key in ("x0_mse", "v_mse")
    }
    bin_ratios: dict[str, dict[str, float]] = {}
    for bin_name, teacher_bin in teacher["bins"].items():
        fake_bin = fake["bins"][bin_name]
        bin_ratios[bin_name] = {
            key: float(fake_bin[key]) / max(float(teacher_bin[key]), 1e-12)
            for key in ("x0_mse", "v_mse")
        }
    aggregate_pass = all(ratio <= 1.0 - improvement_threshold for ratio in ratios.values())
    bins_pass = all(
        ratio <= 1.0 + max_bin_regression
        for values in bin_ratios.values()
        for ratio in values.values()
    )
    return {
        "pass": bool(aggregate_pass and bins_pass),
        "aggregate_pass": bool(aggregate_pass),
        "bins_pass": bool(bins_pass),
        "ratios": ratios,
        "bin_ratios": bin_ratios,
    }


def _load_ema_denoiser(cfg: dict, checkpoint: dict, device: torch.device) -> tuple[Denoiser, object, object]:
    net, model_cfg = build_model(cfg["model"])
    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule).to(device)
    loss_fn = build_loss(cfg["loss"], schedule)
    time_sampler = build_time_sampler(cfg, schedule)
    denoiser = Denoiser(
        net.to(device),
        process=process,
        loss_fn=loss_fn,
        time_sampler=time_sampler,
        time_condition_jitter=cfg.get("time_condition_jitter"),
        model_conditioning=cfg.get("model_conditioning"),
    ).to(device)
    _load_model_state_with_fallback(denoiser, checkpoint["model"], strict=True)
    ema = EMA(denoiser, decay=float((cfg.get("train", {}) or {}).get("ema_decay", 0.996)))
    ema.load_state_dict(_adapt_ema_state_to_model(denoiser, checkpoint["ema"]))
    ema.copy_to(denoiser)
    return denoiser, model_cfg, schedule


@torch.no_grad()
def _fixed_eval_states(
    *,
    fake_x0: torch.Tensor,
    labels: torch.Tensor,
    process: object,
    device: torch.device,
    seed: int,
    min_t: float,
    max_t: float,
) -> dict[str, torch.Tensor]:
    count = fake_x0.shape[0]
    generator = torch.Generator().manual_seed(seed + 17)
    order = torch.randperm(count, generator=generator)
    unit_t = ((torch.arange(count, dtype=torch.float32) + 0.5) / float(count))[order]
    t = min_t + (max_t - min_t) * unit_t
    bin_index = torch.clamp((unit_t * 10).long(), 0, 9)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 29)
        x0 = fake_x0.to(device).float()
        t_device = t.to(device)
        fb = process.forward_sample(x0=x0, t=t_device)
        v_target = process.v_target(fb)
    return {
        "xt": fb.xt.detach().cpu().to(torch.float16),
        "x0": x0.detach().cpu().to(torch.float16),
        "v": v_target.detach().cpu().to(torch.float16),
        "t": t,
        "bin_index": bin_index,
        "labels": labels.detach().cpu().long(),
    }


@torch.inference_mode()
def _evaluate(
    model: Denoiser,
    states: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    per_x0: list[torch.Tensor] = []
    per_v: list[torch.Tensor] = []
    bin_indices: list[torch.Tensor] = []
    pred_rms_sum = 0.0
    target_rms_sum = 0.0
    count = states["x0"].shape[0]
    amp = device.type == "cuda"
    for start in range(0, count, batch_size):
        xt = states["xt"][start : start + batch_size].to(device).float()
        target_x0 = states["x0"][start : start + batch_size].to(device).float()
        target_v = states["v"][start : start + batch_size].to(device).float()
        t = states["t"][start : start + batch_size].to(device)
        cond = states["labels"][start : start + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred_x0 = model(xt, t, cond=cond)
        pred_x0_f = pred_x0.float()
        pred_v = model.process.v_from_output(xt, t, pred_x0_f, aux={}).float()
        per_x0.append((pred_x0_f - target_x0).square().flatten(1).mean(1).cpu())
        per_v.append((pred_v - target_v).square().flatten(1).mean(1).cpu())
        bin_indices.append(states["bin_index"][start : start + batch_size])
        pred_rms_sum += float(pred_x0_f.square().flatten(1).mean(1).sqrt().sum())
        target_rms_sum += float(target_x0.square().flatten(1).mean(1).sqrt().sum())
    x0_error = torch.cat(per_x0)
    v_error = torch.cat(per_v)
    indices = torch.cat(bin_indices)
    bins: dict[str, dict[str, float | int]] = {}
    for index in range(10):
        mask = indices == index
        lo = float(states["t"].min()) + index * float(states["t"].max() - states["t"].min()) / 10.0
        hi = float(states["t"].min()) + (index + 1) * float(states["t"].max() - states["t"].min()) / 10.0
        bins[f"{lo:.3f}-{hi:.3f}"] = {
            "count": int(mask.sum()),
            "x0_mse": float(x0_error[mask].mean()),
            "v_mse": float(v_error[mask].mean()),
        }
    return {
        "x0_mse": float(x0_error.mean()),
        "v_mse": float(v_error.mean()),
        "pred_x0_rms": pred_rms_sum / count,
        "target_x0_rms": target_rms_sum / count,
        "bins": bins,
    }


def _train(
    *,
    args: argparse.Namespace,
    cfg: dict,
    fake_score: Denoiser,
    teacher: Denoiser,
    train_x0: torch.Tensor,
    train_labels: torch.Tensor,
    eval_states: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    for parameter in fake_score.process.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in fake_score.net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    teacher_metrics = _evaluate(teacher, eval_states, device=device, batch_size=args.eval_batch_size)
    initial_metrics = _evaluate(fake_score, eval_states, device=device, batch_size=args.eval_batch_size)
    history: list[dict[str, Any]] = [
        {"step": 0, "train_loss": None, "fake": initial_metrics}
    ]
    n_train = train_x0.shape[0]
    index_generator = torch.Generator().manual_seed(args.seed + 101)
    amp = device.type == "cuda"
    started = time.perf_counter()
    step_times: list[float] = []
    fake_score.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(n_train, (args.batch_size,), generator=index_generator)
        x0 = train_x0[indices].to(device).float()
        cond = train_labels[indices].to(device)
        torch.manual_seed(args.seed + 1009 * step)
        if args.time_distribution == "uniform":
            t = args.min_t + (args.max_t - args.min_t) * torch.rand(args.batch_size, device=device)
        else:
            t = fake_score.time_sampler.sample(args.batch_size, device=device)
        fb = fake_score.process.forward_sample(x0=x0, t=t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = fake_score(fb.xt, t, cond=cond)
            if args.loss_target == "direct_x0":
                loss = (output.float() - x0).square().mean()
            else:
                loss = fake_score.loss_fn(fake_score.process, fb, output)["total"]
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = _evaluate(fake_score, eval_states, device=device, batch_size=args.eval_batch_size)
            row = {"step": step, "train_loss": float(loss.detach()), "fake": metrics}
            history.append(row)
            print(json.dumps(_json_ready(row), sort_keys=True), flush=True)
            fake_score.train()
    duration = time.perf_counter() - started
    final_metrics = history[-1]["fake"]
    decision = readiness_decision(
        teacher_metrics,
        final_metrics,
        improvement_threshold=args.improvement_threshold,
        max_bin_regression=args.max_bin_regression,
    )
    state = {key: value.detach().cpu().clone() for key, value in fake_score.state_dict().items()}
    stable_start = min(len(step_times) - 1, max(10, len(step_times) // 10))
    result = {
        "teacher": teacher_metrics,
        "initial_fake": initial_metrics,
        "history": history,
        "final_fake": final_metrics,
        "decision": decision,
        "duration_s": duration,
        "stable_step_s": float(torch.tensor(step_times[stable_start:]).median()),
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
    }
    return result, state


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    teacher = result["teacher"]
    final = result["final_fake"]
    decision = result["decision"]
    lines = [
        "# Cycle 04 fake-score readiness",
        "",
        f"Frozen terminal distribution: `{result['checkpoint']}`; train={result['num_train']}, "
        f"held-out={result['num_val']}, steps={result['train_steps']}, batch={result['batch_size']}.",
        f"Fake-score target={result['loss_target']}; time={result['time_distribution']} "
        f"[{result['min_t']:.3f}, {result['max_t']:.3f}].",
        "",
        "| model | held-out x0 MSE | held-out v MSE | predicted x0 RMS |",
        "|---|---:|---:|---:|",
        f"| frozen real teacher | {teacher['x0_mse']:.6f} | {teacher['v_mse']:.6f} | {teacher['pred_x0_rms']:.6f} |",
        f"| adapted fake score | {final['x0_mse']:.6f} | {final['v_mse']:.6f} | {final['pred_x0_rms']:.6f} |",
        "",
        f"Ratios fake/teacher: x0={decision['ratios']['x0_mse']:.6f}, "
        f"v={decision['ratios']['v_mse']:.6f}. Aggregate pass={decision['aggregate_pass']}; "
        f"all-bin pass={decision['bins_pass']}; **readiness pass={decision['pass']}**.",
        "",
        f"Runtime={result['duration_s']:.2f}s, stable step={result['stable_step_s']:.6f}s, "
        f"peak memory={result['peak_memory_gib']:.2f} GiB.",
        "",
        "The generator is frozen and receives no gradient in this diagnostic.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.fixed_dataset)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
        raise ValueError("steps, batch-size and eval-every must be positive")
    if not (0.0 <= args.min_t < args.max_t <= 1.0):
        raise ValueError("require 0 <= min-t < max-t <= 1")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint.get("config") or {})
    if not cfg:
        raise ValueError("checkpoint has no embedded config")
    fixed = torch.load(dataset_path, map_location="cpu", weights_only=False)
    n_train = int(fixed["num_train"])
    n_val = int(fixed["num_val"])
    fake_score, _, _ = _load_ema_denoiser(cfg, checkpoint, device)
    teacher = copy.deepcopy(fake_score).to(device).eval()
    eval_states = _fixed_eval_states(
        fake_x0=fixed["fake"][n_train : n_train + n_val],
        labels=fixed["labels"][n_train : n_train + n_val],
        process=fake_score.process,
        device=device,
        seed=args.seed,
        min_t=args.min_t,
        max_t=args.max_t,
    )
    train_result, state = _train(
        args=args,
        cfg=cfg,
        fake_score=fake_score,
        teacher=teacher,
        train_x0=fixed["fake"][:n_train],
        train_labels=fixed["labels"][:n_train],
        eval_states=eval_states,
        device=device,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "fixed_dataset": str(dataset_path),
        "seed": args.seed,
        "num_train": n_train,
        "num_val": n_val,
        "train_steps": args.steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "loss_target": args.loss_target,
        "time_distribution": args.time_distribution,
        "min_t": args.min_t,
        "max_t": args.max_t,
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "thresholds": {
            "improvement": args.improvement_threshold,
            "max_bin_regression": args.max_bin_regression,
        },
        **train_result,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "config": cfg,
            "source_checkpoint": str(checkpoint_path),
            "decision": result["decision"],
        },
        out / "fake_score_step1000.pt",
    )
    (out / "fake_score_readiness.json").write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(_json_ready(result), out / "fake_score_readiness.md")
    print(f"wrote {out / 'fake_score_readiness.json'}", flush=True)


if __name__ == "__main__":
    main()
