from __future__ import annotations

import math
import os
import re

import torch
from PIL import Image, ImageDraw

from x0loop.training.context import LoopConfig, ModelContext, ResumeState, RuntimeContext
from x0loop.utils import dist as dist_utils


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_sample_label_names(cfg: dict) -> tuple[str, ...] | None:
    sample_cfg = cfg.get("sample", {}) or {}
    configured = sample_cfg.get("class_names")
    if configured is not None:
        return tuple(str(name) for name in configured)
    dataset_name = str((cfg.get("dataset", {}) or {}).get("name", "")).lower()
    if dataset_name == "cifar10":
        return CIFAR10_CLASS_NAMES
    return None


def _label_tag(label_id: int, label_names: tuple[str, ...] | None) -> str:
    label = str(label_id)
    if label_names is not None and 0 <= label_id < len(label_names):
        label = label_names[label_id]
    label = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-") or str(label_id)
    return f"_y{label}"


def build_sample_cond(
    cfg: dict,
    *,
    sample_num: int,
    device: torch.device,
    batch_cond: torch.Tensor | None = None,
) -> torch.Tensor | None:
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    if num_classes <= 0:
        return None

    sample_cfg = cfg.get("sample", {})
    labels_cfg = sample_cfg.get("class_labels", None)
    use_batch_cond = bool(sample_cfg.get("use_batch_cond", False))

    cond: torch.Tensor | None = None
    if labels_cfg is not None:
        cond = torch.as_tensor(labels_cfg, device=device, dtype=torch.long).flatten()
    elif use_batch_cond and isinstance(batch_cond, torch.Tensor):
        cond = batch_cond.detach().to(device=device, dtype=torch.long).flatten()
    else:
        # Deterministic default: cycle classes.
        cond = torch.arange(sample_num, device=device, dtype=torch.long) % num_classes

    if cond.numel() == 0:
        return None
    if cond.numel() < sample_num:
        reps = (sample_num + cond.numel() - 1) // cond.numel()
        cond = cond.repeat(reps)
    return cond[:sample_num]


def build_null_class_cond(cfg: dict, *, sample_num: int, device: torch.device) -> torch.Tensor | None:
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    if num_classes <= 0:
        return None
    return torch.full((sample_num,), num_classes, device=device, dtype=torch.long)


def apply_classifier_free_label_dropout(cond: torch.Tensor | None, *, null_class_id: int, drop_prob: float) -> torch.Tensor | None:
    if cond is None or drop_prob <= 0.0:
        return cond
    if not (0.0 <= drop_prob <= 1.0):
        raise ValueError(f"train.class_dropout_prob must be in [0,1], got {drop_prob}")
    drop_mask = torch.rand(cond.shape, device=cond.device) < drop_prob
    return torch.where(drop_mask, torch.full_like(cond, null_class_id), cond)


def save_sample_grid(sample_tensor: torch.Tensor, out_path: str):
    from torchvision.utils import make_grid, save_image

    x = (sample_tensor.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(x, nrow=int(x.shape[0] ** 0.5))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def _tensor_chw_to_uint8_rgb(x: torch.Tensor):
    x = ((x.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c >= 3:
        x = x[:3]
    else:
        pad = torch.zeros((3 - c, h, w), dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    return x.permute(1, 2, 0).contiguous().numpy()


def save_trace_large_images(
    trace: list[dict],
    out_dir: str,
    prefix: str,
    labels: torch.Tensor | None = None,
    label_names: tuple[str, ...] | None = None,
):
    if not trace:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to save trace large images with timestep text.") from exc

    os.makedirs(out_dir, exist_ok=True)
    steps = len(trace)
    bsz = int(trace[0]["x0_hat"].shape[0])
    label_ids = labels.detach().cpu().flatten().tolist() if labels is not None else None
    if label_ids is not None and len(label_ids) != bsz:
        raise ValueError(f"Expected {bsz} sample labels, got {len(label_ids)}.")
    _, h, w = trace[0]["x0_hat"][0].shape
    cols = int(math.ceil(math.sqrt(steps)))
    rows = int(math.ceil(steps / cols))
    pad = 4
    text_h = 14
    cell_w = w + 2 * pad
    cell_h = h + text_h + 2 * pad
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    t_values = [float(item["t"].item()) if torch.is_tensor(item["t"]) else float(item["t"]) for item in trace]

    for bi in range(bsz):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for si, item in enumerate(trace):
            row = si // cols
            col = si % cols
            x0_hat = item["x0_hat"][bi]
            arr = _tensor_chw_to_uint8_rgb(x0_hat)
            img = Image.fromarray(arr)
            x0 = col * cell_w + pad
            y0 = row * cell_h + pad
            canvas.paste(img, (x0, y0))
            draw.text((x0, y0 + h + 1), f"t={t_values[si]:.3f}", fill=(0, 0, 0))

        label_tag = _label_tag(label_ids[bi], label_names) if label_ids is not None else ""
        out_path = os.path.join(out_dir, f"{prefix}_sample_{bi:03d}{label_tag}_x0loop.png")
        canvas.save(out_path)


def run_sampling_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    process,
    ema,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    cond: torch.Tensor | None,
    use_label_cond: bool,
) -> None:
    if loop_cfg.sample_every <= 0 or (resume.global_step % loop_cfg.sample_every != 0):
        return

    should_run_sample = (not loop_cfg.sample_rank0_only) or runtime.is_main
    # FSDP forward is collective; run on all ranks when sharded.
    if model_ctx.use_fsdp and runtime.is_distributed:
        should_run_sample = True

    if should_run_sample:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            sample_num = int(cfg["sample"].get("num", 16))
            sample_cond = build_sample_cond(
                cfg,
                sample_num=sample_num,
                device=runtime.device,
                batch_cond=cond if use_label_cond else None,
            )
            null_cond = build_null_class_cond(cfg, sample_num=sample_num, device=runtime.device)
            guidance_scale = float(cfg["sample"].get("guidance_scale", 1.0))
            sample_sampler = cfg["sample"].get("sampler", None)
            if isinstance(sample_sampler, str) and sample_sampler.lower() == "auto":
                sample_sampler = None
            result = process.sample(
                model=model,
                steps=int(cfg["sample"].get("steps", 50)),
                shape=(sample_num, model_ctx.model_cfg.out_channels, model_ctx.model_cfg.image_size, model_ctx.model_cfg.image_size),
                device=runtime.device,
                dtype=torch.float32,
                return_trace=True,
                cond=sample_cond,
                null_cond=null_cond,
                guidance_scale=guidance_scale,
                sampler=sample_sampler,
                posterior_noise_scale=cfg["sample"].get("posterior_noise_scale", None),
            )
            if ema is not None:
                ema.restore(model)

        if runtime.is_main:
            sample_dir = os.path.join(runtime.out_dir, "samples")
            save_trace_large_images(
                result.get("trace", []),
                sample_dir,
                f"step_{resume.global_step:08d}",
                labels=sample_cond,
                label_names=build_sample_label_names(cfg),
            )
            if bool(cfg["sample"].get("save_trace", False)) and "trace" in result:
                trace_path = os.path.join(sample_dir, f"step_{resume.global_step:08d}_trace.pt")
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                torch.save(result["trace"], trace_path)

        if was_training:
            model.train()

    if runtime.is_distributed:
        dist_utils.barrier()
