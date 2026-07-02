from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any

import torch

from x0loop.core.process_base import BaseProcess
from x0loop.training.context import ModelContext, ResumeState, RuntimeContext
from x0loop.training.sampling import build_null_class_cond, build_sample_label_names, save_sample_images
from x0loop.utils import dist as dist_utils
from x0loop.utils.ema import EMA


def _cfg(cfg: dict) -> dict[str, Any]:
    gen_cfg = dict(cfg.get("gen_eval", {}) or {})
    metric_cfg = dict(gen_cfg.get("metrics", {}) or {})
    sample_cfg = cfg.get("sample", {}) or {}
    out = {
        "enabled": bool(gen_cfg.get("enabled", False)),
        "every_steps": int(gen_cfg.get("every_steps", 10000)),
        "num_samples": int(gen_cfg.get("num_samples", gen_cfg.get("num", 5000))),
        "batch_size": int(gen_cfg.get("batch_size", 128)),
        "steps": int(gen_cfg.get("steps", 20)),
        "sampler": str(gen_cfg.get("sampler", sample_cfg.get("sampler", "heun"))),
        "guidance_scale": float(gen_cfg.get("guidance_scale", 3.0)),
        "guidance_schedule": gen_cfg.get("guidance_schedule", sample_cfg.get("guidance_schedule", None)),
        "time_condition_shift": gen_cfg.get("time_condition_shift", sample_cfg.get("time_condition_shift", None)),
        "posterior_noise_scale": gen_cfg.get("posterior_noise_scale", sample_cfg.get("posterior_noise_scale", None)),
        "input2": gen_cfg.get("input2", None),
        "fid_statistics_file": gen_cfg.get("fid_statistics_file", None),
        "datasets_root": gen_cfg.get("datasets_root", None),
        "datasets_download": bool(gen_cfg.get("datasets_download", False)),
        "keep_images": bool(gen_cfg.get("keep_images", False)),
        "keep_images_count": int(gen_cfg.get("keep_images_count", 0)),
        "verbose": bool(gen_cfg.get("verbose", False)),
        "cache": bool(gen_cfg.get("cache", True)),
        "cache_root": gen_cfg.get("cache_root", None),
        "isc": bool(metric_cfg.get("isc", gen_cfg.get("isc", True))),
        "fid": bool(metric_cfg.get("fid", gen_cfg.get("fid", True))),
        "kid": bool(metric_cfg.get("kid", gen_cfg.get("kid", True))),
        "ppl": bool(metric_cfg.get("ppl", gen_cfg.get("ppl", False))),
        "prc": bool(metric_cfg.get("prc", gen_cfg.get("prc", True))),
        "mind": bool(metric_cfg.get("mind", gen_cfg.get("mind", True))),
    }
    final_cfg = dict(gen_cfg.get("final", {}) or {})
    out["final"] = {
        "enabled": bool(final_cfg.get("enabled", gen_cfg.get("final_enabled", True))),
        "num_samples": int(final_cfg.get("num_samples", gen_cfg.get("final_num_samples", 20000))),
        "batch_size": int(final_cfg.get("batch_size", gen_cfg.get("final_batch_size", out["batch_size"]))),
        "steps": int(final_cfg.get("steps", gen_cfg.get("final_steps", 50))),
            "sampler": str(final_cfg.get("sampler", gen_cfg.get("final_sampler", out["sampler"]))),
            "guidance_scale": float(final_cfg.get("guidance_scale", gen_cfg.get("final_guidance_scale", out["guidance_scale"]))),
            "guidance_schedule": final_cfg.get("guidance_schedule", gen_cfg.get("final_guidance_schedule", out["guidance_schedule"])),
            "time_condition_shift": final_cfg.get("time_condition_shift", gen_cfg.get("final_time_condition_shift", out["time_condition_shift"])),
        }
    return out


def _default_input2(cfg: dict) -> str | None:
    dataset_name = str((cfg.get("dataset", {}) or {}).get("name", "")).lower()
    if dataset_name == "cifar10":
        return "cifar10-train"
    return None


def _default_datasets_root(cfg: dict, input2: str | None) -> str | None:
    if input2 not in {"cifar10-train", "cifar10-val"}:
        return None
    root = (cfg.get("dataset", {}) or {}).get("root", None)
    return str(root) if root else None


def _labels_for_indices(cfg: dict, indices: list[int], device: torch.device) -> torch.Tensor | None:
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    if num_classes <= 0:
        return None
    return (torch.as_tensor(indices, device=device, dtype=torch.long) % num_classes).flatten()


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _headline_metrics(metrics: dict) -> dict[str, Any]:
    """Pull the most-watched scores to the front of each row for at-a-glance reading."""
    priority = (
        "frechet_inception_distance",
        "inception_score_mean",
        "kernel_inception_distance_mean",
        "precision",
        "recall",
        "f_score",
    )
    return {k: metrics[k] for k in priority if k in metrics}


def _metrics_path(runtime: RuntimeContext) -> str:
    timestamp = getattr(runtime.logger, "run_timestamp", time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    return os.path.join(runtime.out_dir, f"gen_eval_metrics_{timestamp}.jsonl")


def _write_metrics_jsonl(runtime: RuntimeContext, row: dict[str, Any]) -> str:
    path = _metrics_path(runtime)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_ready(row), ensure_ascii=True) + "\n")
    return path


def _prune_fake_images(fake_dir: str, keep_count: int) -> int:
    if keep_count <= 0:
        return 0
    paths = []
    for name in os.listdir(fake_dir):
        path = os.path.join(fake_dir, name)
        if os.path.isfile(path):
            paths.append(path)
    paths.sort()
    for path in paths[keep_count:]:
        os.remove(path)
    return min(len(paths), keep_count)


@contextmanager
def _torch_fidelity_load_compat():
    """Make older torch-fidelity cache loads work on PyTorch 2.6+.

    torch-fidelity 0.3/0.4 calls torch.load(cache_path) for numpy-based feature
    caches without passing weights_only. PyTorch 2.6 changed the default to
    weights_only=True, which rejects those local cache files. The cache is
    generated by torch-fidelity itself, so use the old behavior only while
    calculate_metrics is running.
    """

    original_load = torch.load

    def compat_load(*args, **kwargs):
        if kwargs.get("weights_only", None) is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = compat_load
    try:
        yield
    finally:
        torch.load = original_load


@torch.no_grad()
def _export_fake_images(
    *,
    cfg: dict,
    gen_cfg: dict[str, Any],
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process: BaseProcess,
    fake_dir: str,
) -> None:
    local_indices = list(range(runtime.rank, int(gen_cfg["num_samples"]), runtime.world_size))
    label_names = build_sample_label_names(cfg)
    null_cond_full = build_null_class_cond(cfg, sample_num=int(gen_cfg["batch_size"]), device=runtime.device)

    for offset in range(0, len(local_indices), int(gen_cfg["batch_size"])):
        indices = local_indices[offset : offset + int(gen_cfg["batch_size"])]
        cond = _labels_for_indices(cfg, indices, runtime.device)
        null_cond = null_cond_full[: len(indices)] if null_cond_full is not None else None
        result = process.sample(
            model=model,
            steps=int(gen_cfg["steps"]),
            shape=(len(indices), model_ctx.model_cfg.in_channels, model_ctx.model_cfg.image_size, model_ctx.model_cfg.image_size),
            device=runtime.device,
            dtype=torch.float32,
            return_trace=False,
            cond=cond,
            null_cond=null_cond,
            guidance_scale=float(gen_cfg["guidance_scale"]),
            guidance_schedule=gen_cfg["guidance_schedule"],
            time_condition_shift=gen_cfg["time_condition_shift"],
            sampler=str(gen_cfg["sampler"]),
            posterior_noise_scale=gen_cfg["posterior_noise_scale"],
        )
        labels_cpu = cond.detach().cpu() if cond is not None else None
        save_sample_images(result["x"].detach().cpu(), fake_dir, indices=indices, labels=labels_cpu, label_names=label_names, cfg=cfg)


def _calculate_metrics(cfg: dict, gen_cfg: dict[str, Any], fake_dir: str, runtime: RuntimeContext) -> dict:
    from torch_fidelity import calculate_metrics

    input2 = gen_cfg["input2"] or _default_input2(cfg)
    metric_kwargs: dict[str, Any] = {
        "input1": fake_dir,
        "cuda": bool(runtime.device.type == "cuda"),
        "isc": bool(gen_cfg["isc"]),
        "fid": bool(gen_cfg["fid"]),
        "kid": bool(gen_cfg["kid"]),
        "ppl": bool(gen_cfg["ppl"]),
        "prc": bool(gen_cfg["prc"]),
        "mind": bool(gen_cfg["mind"]),
        "verbose": bool(gen_cfg["verbose"]),
        "cache": bool(gen_cfg["cache"]),
    }
    if input2 is not None:
        metric_kwargs["input2"] = input2
    if gen_cfg["fid_statistics_file"] is not None:
        metric_kwargs["fid_statistics_file"] = gen_cfg["fid_statistics_file"]
    if gen_cfg["cache_root"]:
        metric_kwargs["cache_root"] = gen_cfg["cache_root"]
    datasets_root = gen_cfg["datasets_root"] or _default_datasets_root(cfg, input2)
    if datasets_root is not None:
        metric_kwargs["datasets_root"] = datasets_root
    metric_kwargs["datasets_download"] = bool(gen_cfg["datasets_download"])
    if metric_kwargs["fid"] and input2 is None and not gen_cfg["fid_statistics_file"]:
        raise ValueError("gen_eval requires input2 or fid_statistics_file for FID unless dataset.name=cifar10.")
    with _torch_fidelity_load_compat():
        return calculate_metrics(**metric_kwargs)


def _run_generative_eval(
    *,
    cfg: dict,
    gen_cfg: dict[str, Any],
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process: BaseProcess,
    ema: EMA | None,
    resume: ResumeState,
    tag: str,
) -> None:
    if int(gen_cfg["num_samples"]) <= 0:
        raise ValueError(f"gen_eval.num_samples must be > 0, got {gen_cfg['num_samples']}")

    eval_dir = os.path.join(runtime.out_dir, "gen_eval", tag)
    fake_dir = os.path.join(eval_dir, "fake")
    if runtime.is_main:
        os.makedirs(fake_dir, exist_ok=True)
        runtime.logger.log_text(
            "[gen_eval] start: "
            f"tag={tag}, step={resume.global_step}, num_samples={gen_cfg['num_samples']}, "
            f"steps={gen_cfg['steps']}, sampler={gen_cfg['sampler']}, "
            f"guidance_scale={gen_cfg['guidance_scale']}, guidance_schedule={gen_cfg['guidance_schedule']}, "
            f"time_condition_shift={gen_cfg['time_condition_shift']}, "
            f"model_conditioning={cfg.get('model_conditioning', {'ignore_time': False})}"
        )
    if runtime.is_distributed:
        dist_utils.barrier()

    was_training = model.training
    model.eval()
    gen_ema = ema
    if model_ctx.use_fsdp and ema is not None:
        gen_ema = None
        if runtime.is_main:
            runtime.logger.log_text("[gen_eval] EMA skipped for FSDP/DTensor model; using current weights.")
    if gen_ema is not None:
        gen_ema.store(model)
        gen_ema.copy_to(model)
    try:
        _export_fake_images(cfg=cfg, gen_cfg=gen_cfg, model=model, runtime=runtime, model_ctx=model_ctx, process=process, fake_dir=fake_dir)
    finally:
        if gen_ema is not None:
            gen_ema.restore(model)
        if was_training:
            model.train()
    if runtime.is_distributed:
        dist_utils.barrier()

    if runtime.is_main:
        metrics: dict = {}
        error: str | None = None
        try:
            metrics = _calculate_metrics(cfg, gen_cfg, fake_dir, runtime)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            runtime.logger.log_text(f"[gen_eval] metrics_failed tag={tag}: {error}")
        row = {
            **_headline_metrics(metrics),
            "step": int(resume.global_step),
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "tag": tag,
            "eval_dir": eval_dir,
            "fake_dir": fake_dir if gen_cfg["keep_images"] or int(gen_cfg["keep_images_count"]) > 0 else None,
            "num_samples": int(gen_cfg["num_samples"]),
            "batch_size": int(gen_cfg["batch_size"]),
            "steps": int(gen_cfg["steps"]),
            "sampler": str(gen_cfg["sampler"]),
            "guidance_scale": float(gen_cfg["guidance_scale"]),
            "guidance_schedule": gen_cfg["guidance_schedule"],
            "time_condition_shift": gen_cfg["time_condition_shift"],
            "model_conditioning": cfg.get("model_conditioning", {"ignore_time": False}),
            "keep_images_count": int(gen_cfg["keep_images_count"]),
            "metrics": metrics,
        }
        if error is not None:
            row["error"] = error
        metrics_path = _write_metrics_jsonl(runtime, row)
        runtime.logger.log_text(f"[gen_eval] metrics_jsonl={metrics_path}")
        if error is None:
            runtime.logger.log_text(f"[gen_eval] metrics={_json_ready(metrics)}")
        keep_count = int(gen_cfg["keep_images_count"])
        if keep_count > 0:
            kept = _prune_fake_images(fake_dir, keep_count)
            runtime.logger.log_text(f"[gen_eval] kept_fake_images={kept} fake_dir={fake_dir}")
        elif not gen_cfg["keep_images"]:
            shutil.rmtree(eval_dir, ignore_errors=True)
    if runtime.is_distributed:
        dist_utils.barrier()


def run_generative_eval_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process: BaseProcess,
    ema: EMA | None,
    resume: ResumeState,
) -> float | None:
    gen_cfg = _cfg(cfg)
    if not gen_cfg["enabled"]:
        return None
    every_steps = int(gen_cfg["every_steps"])
    if every_steps <= 0 or resume.global_step <= 0 or (resume.global_step % every_steps != 0):
        return None
    start = time.time()
    _run_generative_eval(
        cfg=cfg,
        gen_cfg=gen_cfg,
        model=model,
        runtime=runtime,
        model_ctx=model_ctx,
        process=process,
        ema=ema,
        resume=resume,
        tag=f"step_{resume.global_step:08d}",
    )
    return time.time() - start


def run_final_generative_eval(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process: BaseProcess,
    ema: EMA | None,
    resume: ResumeState,
) -> float | None:
    gen_cfg = _cfg(cfg)
    final_cfg = dict(gen_cfg.pop("final"))
    if not gen_cfg["enabled"] or not final_cfg["enabled"]:
        return None
    gen_cfg.update(final_cfg)
    start = time.time()
    _run_generative_eval(
        cfg=cfg,
        gen_cfg=gen_cfg,
        model=model,
        runtime=runtime,
        model_ctx=model_ctx,
        process=process,
        ema=ema,
        resume=resume,
        tag=f"final_step_{resume.global_step:08d}",
    )
    return time.time() - start
