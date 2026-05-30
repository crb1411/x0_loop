"""Stand-alone inference script.

Config resolution order:
  1. <ckpt_dir>/../resolved_config.yaml   (auto-detected from checkpoint path)
  2. infer/infer.yaml                      (overlaid on top, overrides anything)

Outputs (under <ckpt_dir>/infer_step_XXXXXXXX/ by default):
  step_XXXXXXXX_sample_NNN_x0loop.png   — per-sample denoising trace
  sample_grid.png                        — flat grid of all samples
  infer_config.yaml                      — full resolved config for this run
"""
from __future__ import annotations

import argparse
import copy
import os
import re

import math

import numpy as np
import torch
import yaml

from x0loop.models.dit import DiT, DiTConfig
from x0loop.train import (
    build_null_class_cond,
    build_process,
    build_sample_cond,
    build_schedule,
    save_sample_grid,
    save_trace_large_images,
)
from x0loop.utils.checkpoint import load_checkpoint
from x0loop.utils.ema import EMA


def _save_grid_labeled(
    images: torch.Tensor,   # [N, C, H, W] float in [-1, 1]
    labels: list[str],
    out_path: str,
    upscale: int = 4,
) -> None:
    """Save a grid where each image has its class label printed below it."""
    from PIL import Image, ImageDraw

    N, C, H, W = images.shape
    # layout: single row when N≤8, else square-ish grid
    imgs_per_row = N if N <= 8 else math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / imgs_per_row)

    cell_w  = W * upscale
    cell_h  = H * upscale
    label_h = 16
    gap     = 4

    canvas_w = imgs_per_row * cell_w + (imgs_per_row + 1) * gap
    canvas_h = n_rows * (cell_h + label_h) + (n_rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    imgs_u8 = ((images.clamp(-1, 1) + 1.0) * 127.5).byte().cpu()

    for idx in range(N):
        col = idx % imgs_per_row
        row = idx // imgs_per_row
        x0  = gap + col * (cell_w + gap)
        y0  = gap + row * (cell_h + label_h + gap)

        arr = imgs_u8[idx].permute(1, 2, 0).numpy()   # HWC uint8
        if C == 1:
            arr = np.repeat(arr, 3, axis=2)
        pil = Image.fromarray(arr[:, :, :3]).resize((cell_w, cell_h), Image.NEAREST)
        canvas.paste(pil, (x0, y0))

        label = labels[idx] if idx < len(labels) else f"cls {idx}"
        # center the text under the image
        bbox  = draw.textbbox((0, 0), label)
        tw    = bbox[2] - bbox[0]
        tx    = x0 + max(0, (cell_w - tw) // 2)
        draw.text((tx, y0 + cell_h + 2), label, fill=(220, 220, 80))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def _parse_args():
    p = argparse.ArgumentParser(description="x0loop inference")
    p.add_argument("--ckpt",         type=str, required=True,
                   help="checkpoint path (.pt)")
    p.add_argument("--infer-config", type=str, default="infer/infer.yaml",
                   help="inference overlay yaml (default: infer/infer.yaml)")
    p.add_argument("--out",          type=str, default=None,
                   help="output dir (default: <ckpt_dir>/infer_step_XXXXXXXX/)")
    p.add_argument("--device",       type=str, default=None,
                   help="e.g. cuda:0  (default: cuda if available)")
    return p.parse_args()


def _step_from_ckpt(ckpt_path: str) -> int:
    m = re.search(r"ckpt_step_0*(\d+)", os.path.basename(ckpt_path))
    return int(m.group(1)) if m else 0


def _find_resolved_config(ckpt_path: str) -> str:
    """Walk up from checkpoints/ to find resolved_config.yaml in the run dir."""
    ckpt_abs = os.path.abspath(ckpt_path)
    # checkpoints/ is one level below the run dir
    run_dir = os.path.dirname(os.path.dirname(ckpt_abs))
    candidate = os.path.join(run_dir, "resolved_config.yaml")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"resolved_config.yaml not found at {candidate}\n"
        f"Expected layout: <run_dir>/resolved_config.yaml + <run_dir>/checkpoints/<ckpt>.pt"
    )


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model_cfg = DiTConfig(**cfg["model"])
    model = DiT(model_cfg).to(device)
    use_ema = bool(cfg["train"].get("use_ema", True))
    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.9999))) if use_ema else None
    ckpt_mode = cfg.get("distributed", {}).get("checkpoint", {}).get("mode", "full")
    try:
        load_checkpoint(ckpt_path, model=model, optimizer=None,
                        scaler=None, ema=ema, map_location="cpu", mode=ckpt_mode, strict=True)
    except Exception:
        info = load_checkpoint(ckpt_path, model=model, optimizer=None,
                               scaler=None, ema=ema, map_location="cpu", mode="full", strict=False)
        li = info.get("_load_info", {})
        print(f"[infer] fallback strict=False; "
              f"missing={len(li.get('missing_keys', []))}, "
              f"unexpected={len(li.get('unexpected_keys', []))}", flush=True)
    if ema is not None:
        ema.copy_to(model)
    return model, model_cfg


def _run_tag(sc: dict) -> str:
    """Build a short tag encoding key sample settings for directory naming."""
    guidance = float(sc.get("guidance_scale", 1.0))
    sampler  = str(sc.get("sampler", "ddim")).lower()
    if sampler in ("auto", ""):
        sampler = "ddim"
    num   = int(sc.get("num", 16))
    steps = int(sc.get("steps", 100))

    # guidance: 1.0 → cfg1.0, 1.5 → cfg1.5
    gs = f"cfg{guidance:.1f}"
    # sampler: shorten posterior → post
    sm = "post" if sampler == "posterior" else sampler
    return f"{gs}_{sm}_n{num}_t{steps}"


def _dump_config(cfg: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "infer_config.yaml")
    skip = {"_config_path", "_runtime_config_path"}
    clean = {k: v for k, v in cfg.items() if k not in skip}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
    print(f"[infer] config  → {path}", flush=True)


def main():
    args = _parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Config resolution ──────────────────────────────────────────────────
    resolved_cfg_path = _find_resolved_config(args.ckpt)
    print(f"[infer] base config → {resolved_cfg_path}", flush=True)
    cfg = _load_yaml(resolved_cfg_path)

    if os.path.isfile(args.infer_config):
        print(f"[infer] overlay     → {args.infer_config}", flush=True)
        cfg = _deep_merge(cfg, _load_yaml(args.infer_config))
    else:
        print(f"[infer] no overlay found at {args.infer_config}, using base config only", flush=True)

    # ── Sampling params (resolved here so run_tag is accurate) ────────────
    sc           = cfg.get("sample", {})
    sample_num   = int(sc.get("num", 16))
    steps        = int(sc.get("steps", 100))
    guidance     = float(sc.get("guidance_scale", 1.0))
    return_trace = bool(sc.get("save_trace", True))
    sampler_name = sc.get("sampler", "ddim")
    if isinstance(sampler_name, str) and sampler_name.lower() == "auto":
        sampler_name = None
    posterior_ns = sc.get("posterior_noise_scale", None)

    # ── Output directory: infer_step_XXXXXXXX / cfg{g}_{sm}_n{num}_t{steps} ──
    step     = _step_from_ckpt(args.ckpt)
    step_tag = f"step_{step:08d}"
    run_tag  = _run_tag(sc)

    if args.out:
        out_dir = args.out
    else:
        ckpt_dir   = os.path.dirname(os.path.abspath(args.ckpt))
        infer_root = os.path.join(ckpt_dir, f"infer_{step_tag}")
        out_dir    = os.path.join(infer_root, run_tag)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[infer] output dir  → {out_dir}", flush=True)

    _dump_config(cfg, out_dir)

    # ── Model ──────────────────────────────────────────────────────────────
    model, model_cfg = _load_model(cfg, args.ckpt, device)
    model.eval()

    schedule = build_schedule(cfg)
    process  = build_process(cfg, schedule)

    # ── Sampling ───────────────────────────────────────────────────────────
    sample_cond = build_sample_cond(cfg, sample_num=sample_num, device=device, batch_cond=None)
    null_cond   = build_null_class_cond(cfg, sample_num=sample_num, device=device)

    print(f"[infer] generating {sample_num} images × {steps} steps  "
          f"guidance={guidance}  sampler={sampler_name or 'ddim'}", flush=True)

    with torch.no_grad():
        result = process.sample(
            model=model,
            steps=steps,
            shape=(sample_num, model_cfg.out_channels, model_cfg.image_size, model_cfg.image_size),
            device=device,
            dtype=torch.float32,
            return_trace=return_trace,
            cond=sample_cond,
            null_cond=null_cond,
            guidance_scale=guidance,
            sampler=sampler_name,
            posterior_noise_scale=posterior_ns,
        )

    # ── Save ───────────────────────────────────────────────────────────────
    if return_trace and result.get("trace"):
        save_trace_large_images(result["trace"], out_dir, prefix=step_tag)
        print(f"[infer] trace PNGs  → {out_dir}/{step_tag}_sample_*.png", flush=True)

    grid_path = os.path.join(out_dir, "sample_grid.png")
    imgs_cpu  = result["x"].detach().cpu()

    if guidance > 1.0 and sample_cond is not None:
        # Build human-readable label strings from class IDs
        class_names = cfg.get("sample", {}).get("class_names") or []
        cond_ids = sample_cond.cpu().tolist()
        label_strs = [
            class_names[cid] if class_names and cid < len(class_names) else f"cls {cid}"
            for cid in cond_ids
        ]
        _save_grid_labeled(imgs_cpu, label_strs, grid_path)
        print(f"[infer] grid PNG (labeled cfg={guidance}) → {grid_path}", flush=True)
    else:
        save_sample_grid(imgs_cpu, grid_path)
        print(f"[infer] grid PNG    → {grid_path}", flush=True)


if __name__ == "__main__":
    main()
