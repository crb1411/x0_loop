from __future__ import annotations

import math

import torch


def amp_dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def maybe_make_scaler(precision: str, use_fsdp: bool):
    # use_fsdp kept for API compatibility; GradScaler is unified in torch.amp.
    del use_fsdp
    if precision != "fp16" or not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        # Backward-compatible path for older torch.amp signatures.
        return torch.amp.GradScaler()


def build_step_lr_schedule(train_cfg: dict, total_steps: int, steps_per_epoch: int):
    lr_cfg = train_cfg.get("lr_scheduler", {}) or {}
    name = str(lr_cfg.get("name", "constant")).lower()
    base_lr = float(train_cfg.get("lr", 1e-4))

    if name in {"none", "constant"}:
        def _const_lr(step: int) -> float:
            del step
            return base_lr
        return _const_lr, {"name": "constant", "base_lr": base_lr}

    if name != "cosine":
        raise ValueError(f"Unsupported train.lr_scheduler.name={name}, use constant|cosine")

    max_lr = float(lr_cfg.get("max_lr", base_lr))
    min_lr = float(lr_cfg.get("min_lr", 0.0))

    # New preferred schema:
    # 1) warmup from min_lr -> max_lr for warmup_steps
    # 2) cosine from max_lr -> min_lr for cosine_steps
    # 3) hold min_lr afterwards
    if "warmup_steps" in lr_cfg or "cosine_steps" in lr_cfg:
        warmup_steps = int(lr_cfg.get("warmup_steps", 0))
        cosine_steps = int(lr_cfg.get("cosine_steps", 0))
        for n, v in {"warmup_steps": warmup_steps, "cosine_steps": cosine_steps}.items():
            if v < 0:
                raise ValueError(f"train.lr_scheduler.{n} must be >= 0, got {v}")

        def _warmup_cosine_lr(step: int) -> float:
            s = int(step)
            if warmup_steps > 0 and s < warmup_steps:
                # Linear warmup from min -> max.
                if warmup_steps == 1:
                    return max_lr
                p = float(s) / float(warmup_steps - 1)
                return min_lr + p * (max_lr - min_lr)

            if cosine_steps > 0 and s < warmup_steps + cosine_steps:
                # Cosine decay from max -> min.
                if cosine_steps == 1:
                    return min_lr
                k = s - warmup_steps
                p = float(k) / float(cosine_steps - 1)
                p = min(max(p, 0.0), 1.0)
                return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))

            return min_lr

        meta = {
            "name": "cosine_warmup_hold",
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "cosine_steps": cosine_steps,
            "hold_min_from_step": warmup_steps + cosine_steps,
        }
        return _warmup_cosine_lr, meta

    # Backward-compatible schema:
    # init_steps (init_lr hold) -> max_steps (max_lr hold) -> cosine -> min_steps (min_lr hold)
    init_lr = float(lr_cfg.get("init_lr", base_lr))
    if any(k in lr_cfg for k in ("init_steps", "max_steps", "min_steps")):
        init_steps = int(lr_cfg.get("init_steps", 0))
        max_steps = int(lr_cfg.get("max_steps", 0))
        min_steps = int(lr_cfg.get("min_steps", 0))
    else:
        init_steps = int(lr_cfg.get("init_epochs", 0)) * int(steps_per_epoch)
        max_steps = int(lr_cfg.get("max_epochs", 0)) * int(steps_per_epoch)
        min_steps = int(lr_cfg.get("min_epochs", 0)) * int(steps_per_epoch)

    for n, v in {"init_steps": init_steps, "max_steps": max_steps, "min_steps": min_steps}.items():
        if v < 0:
            raise ValueError(f"train.lr_scheduler.{n} must be >= 0, got {v}")
    occupied = init_steps + max_steps + min_steps
    if occupied > total_steps:
        raise ValueError(
            "train.lr_scheduler init_steps+max_steps+min_steps must be <= total_train_steps, "
            f"got {occupied} > {total_steps}"
        )

    cosine_start = init_steps + max_steps
    cosine_end = total_steps - min_steps
    cosine_steps = max(0, cosine_end - cosine_start)

    def _cosine_lr_legacy(step: int) -> float:
        s = int(step)
        if s < init_steps:
            return init_lr
        if s < cosine_start:
            return max_lr
        if s >= cosine_end:
            return min_lr
        if cosine_steps <= 0:
            return min_lr
        p = float(s - cosine_start + 1) / float(cosine_steps)
        p = min(max(p, 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))

    meta = {
        "name": "cosine_legacy",
        "init_lr": init_lr,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "init_steps": init_steps,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "cosine_steps": cosine_steps,
    }
    return _cosine_lr_legacy, meta


def maybe_compile_model(model: torch.nn.Module, compile_cfg: dict):
    if not bool(compile_cfg.get("enabled", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(
        model,
        mode=str(compile_cfg.get("mode", "default")),
        fullgraph=bool(compile_cfg.get("fullgraph", False)),
        dynamic=bool(compile_cfg.get("dynamic", False)),
    )
