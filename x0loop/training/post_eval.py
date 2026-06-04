from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any

import torch
import yaml

from x0loop.core.process_base import BaseProcess
from x0loop.training.context import ModelContext, ResumeState, RuntimeContext
from x0loop.training.sampling import build_null_class_cond, build_sample_cond, build_sample_label_names, save_sample_grid, save_sample_images
from x0loop.utils import dist as dist_utils
from x0loop.utils.ema import EMA


def _post_eval_cfg(cfg: dict) -> dict:
    post_cfg = dict(cfg.get("post_eval", {}) or {})
    return {
        "enabled": bool(post_cfg.get("enabled", False)),
        "steps": int(post_cfg.get("steps", cfg.get("sample", {}).get("steps", 50))),
        "num": int(post_cfg.get("num", cfg.get("sample", {}).get("num", 5))),
        "batch_size": int(post_cfg.get("batch_size", post_cfg.get("num", cfg.get("sample", {}).get("num", 5)))),
        "sampler": str(post_cfg.get("sampler", "heun")),
        "guidance_scale": float(post_cfg.get("guidance_scale", 3.0)),
        "posterior_noise_scale": post_cfg.get("posterior_noise_scale", cfg.get("sample", {}).get("posterior_noise_scale", None)),
        "save_images": bool(post_cfg.get("save_images", True)),
        "save_grid": bool(post_cfg.get("save_grid", True)),
        "out_dir": post_cfg.get("out_dir", None),
        "class_labels": post_cfg.get("class_labels", None),
        "class_names": post_cfg.get("class_names", None),
    }


def _cfg_for_labels(cfg: dict, post_cfg: dict) -> dict:
    label_cfg = deepcopy(cfg)
    sample_cfg = dict(label_cfg.get("sample", {}) or {})
    for key in ("class_labels", "class_names"):
        if post_cfg.get(key) is not None:
            sample_cfg[key] = post_cfg[key]
    sample_cfg["use_batch_cond"] = False
    label_cfg["sample"] = sample_cfg
    return label_cfg


def _yaml_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _yaml_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_ready(v) for v in value]
    return value


@torch.no_grad()
def run_post_train_eval(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process: BaseProcess,
    ema: EMA | None,
    resume: ResumeState,
) -> None:
    post_cfg = _post_eval_cfg(cfg)
    if not post_cfg["enabled"]:
        return
    if post_cfg["num"] <= 0:
        raise ValueError(f"post_eval.num must be > 0, got {post_cfg['num']}")
    if post_cfg["batch_size"] <= 0:
        raise ValueError(f"post_eval.batch_size must be > 0, got {post_cfg['batch_size']}")

    should_run = runtime.is_main
    if model_ctx.use_fsdp and runtime.is_distributed:
        should_run = True
    if not should_run:
        if runtime.is_distributed:
            dist_utils.barrier()
        return

    label_cfg = _cfg_for_labels(cfg, post_cfg)
    sample_num = int(post_cfg["num"])
    batch_size = min(int(post_cfg["batch_size"]), sample_num)
    label_names = build_sample_label_names(label_cfg)
    sample_cond = build_sample_cond(label_cfg, sample_num=sample_num, device=runtime.device, batch_cond=None)
    null_cond = build_null_class_cond(label_cfg, sample_num=sample_num, device=runtime.device)

    out_dir = str(post_cfg["out_dir"] or os.path.join(runtime.out_dir, "post_eval"))
    image_dir = os.path.join(out_dir, "images")
    grid_path = os.path.join(out_dir, "grid.png")
    yaml_path = os.path.join(out_dir, "post_eval.yaml")
    os.makedirs(out_dir, exist_ok=True)

    if runtime.is_main:
        runtime.logger.log_text(
            "[post_eval] start: "
            f"num={sample_num}, steps={post_cfg['steps']}, sampler={post_cfg['sampler']}, "
            f"guidance_scale={post_cfg['guidance_scale']}, out_dir={out_dir}"
        )

    was_training = model.training
    model.eval()
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)

    samples: list[torch.Tensor] = []
    image_paths: list[str] = []
    try:
        for start in range(0, sample_num, batch_size):
            end = min(start + batch_size, sample_num)
            current = end - start
            cond_batch = sample_cond[start:end] if sample_cond is not None else None
            null_batch = null_cond[start:end] if null_cond is not None else None
            result = process.sample(
                model=model,
                steps=int(post_cfg["steps"]),
                shape=(current, model_ctx.model_cfg.out_channels, model_ctx.model_cfg.image_size, model_ctx.model_cfg.image_size),
                device=runtime.device,
                dtype=torch.float32,
                return_trace=False,
                cond=cond_batch,
                null_cond=null_batch,
                guidance_scale=float(post_cfg["guidance_scale"]),
                sampler=str(post_cfg["sampler"]),
                posterior_noise_scale=post_cfg["posterior_noise_scale"],
            )
            batch_x = result["x"].detach().cpu()
            samples.append(batch_x)
            if runtime.is_main and post_cfg["save_images"]:
                labels_cpu = cond_batch.detach().cpu() if cond_batch is not None else None
                image_paths.extend(save_sample_images(batch_x, image_dir, start_index=start, labels=labels_cpu, label_names=label_names))
    finally:
        if ema is not None:
            ema.restore(model)
        if was_training:
            model.train()

    if runtime.is_main:
        all_samples = torch.cat(samples, dim=0)
        if post_cfg["save_grid"]:
            save_sample_grid(all_samples, grid_path)

        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "step": int(resume.global_step),
            "run_dir": runtime.out_dir,
            "post_eval_dir": out_dir,
            "dataset": cfg.get("dataset", {}),
            "model": cfg.get("model", {}),
            "process": cfg.get("process", {}),
            "schedule": cfg.get("schedule", {}),
            "time_sampler": cfg.get("time_sampler", {}),
            "loss": cfg.get("loss", {}),
            "sampling": {
                "steps": int(post_cfg["steps"]),
                "num": sample_num,
                "batch_size": batch_size,
                "sampler": str(post_cfg["sampler"]),
                "guidance_scale": float(post_cfg["guidance_scale"]),
                "posterior_noise_scale": post_cfg["posterior_noise_scale"],
                "class_labels": sample_cond.detach().cpu().tolist() if sample_cond is not None else None,
                "class_names": list(label_names) if label_names is not None else None,
            },
            "artifacts": {
                "images_dir": image_dir if post_cfg["save_images"] else None,
                "num_images": len(image_paths),
                "grid": grid_path if post_cfg["save_grid"] else None,
            },
            "metrics": {},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_yaml_ready(metadata), f, sort_keys=False)
        runtime.logger.log_text(f"[post_eval] yaml={yaml_path}")
        runtime.logger.log_kv(
            int(resume.global_step),
            {
                "post_eval/num": float(sample_num),
                "post_eval/steps": float(post_cfg["steps"]),
                "post_eval/guidance_scale": float(post_cfg["guidance_scale"]),
            },
        )

    if runtime.is_distributed:
        dist_utils.barrier()
