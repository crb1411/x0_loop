from __future__ import annotations

import argparse
import os

import torch

from x0loop.core.config import DEFAULT_RUNTIME_CONFIG, load_merged_config
from x0loop.models.factory import build_model
from x0loop.train import build_null_class_cond, build_process, build_sample_cond, build_schedule, save_sample_grid
from x0loop.utils.checkpoint import load_checkpoint
from x0loop.utils.dist import init_distributed, is_main_process
from x0loop.utils.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="x0loop/configs/default.yaml")
    p.add_argument("--runtime-config", type=str, default=DEFAULT_RUNTIME_CONFIG)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--sampler", type=str, default="auto", choices=["auto", "ddim", "posterior", "euler", "heun"])
    p.add_argument("--posterior-noise-scale", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config(args.config, args.runtime_config)

    backend = cfg.get("distributed", {}).get("backend", "nccl")
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
    dist_info = init_distributed(backend=backend)

    local_rank = dist_info["local_rank"]
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = build_model(cfg["model"])
    model = model.to(device)

    ema = EMA(model, decay=float(cfg["train"].get("ema_decay", 0.9999))) if bool(cfg["train"].get("use_ema", True)) else None
    ckpt_mode = cfg.get("distributed", {}).get("checkpoint", {}).get("mode", "full")
    try:
        ckpt = load_checkpoint(
            args.ckpt,
            model=model,
            optimizer=None,
            scaler=None,
            ema=ema,
            map_location="cpu",
            mode=ckpt_mode,
            strict=True,
        )
    except Exception:
        # Common fallback when a sharded export is consumed from a non-FSDP sampling path.
        ckpt = load_checkpoint(
            args.ckpt,
            model=model,
            optimizer=None,
            scaler=None,
            ema=ema,
            map_location="cpu",
            mode="full",
            strict=False,
        )
        if is_main_process():
            info = ckpt.get("_load_info", {})
            missing = info.get("missing_keys", [])
            unexpected = info.get("unexpected_keys", [])
            prefix = info.get("prefix", "none")
            print(
                f"[sample][ckpt] fallback strict=False used; prefix={prefix}; "
                f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(f"[sample][ckpt] missing_keys: {missing}", flush=True)
            if unexpected:
                print(f"[sample][ckpt] unexpected_keys: {unexpected}", flush=True)

    if ema is not None:
        ema.copy_to(model)

    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule)

    model.eval()
    sample_num = int(cfg["sample"].get("num", 16))
    sample_cond = build_sample_cond(cfg, sample_num=sample_num, device=device, batch_cond=None)
    null_cond = build_null_class_cond(cfg, sample_num=sample_num, device=device)
    with torch.no_grad():
        sampler = None if args.sampler == "auto" else args.sampler
        if sampler is None:
            cfg_sampler = cfg["sample"].get("sampler", None)
            if isinstance(cfg_sampler, str) and cfg_sampler.lower() == "auto":
                cfg_sampler = None
            sampler = cfg_sampler
        result = process.sample(
            model=model,
            steps=int(args.steps if args.steps is not None else cfg["sample"].get("steps", 50)),
            shape=(sample_num, model_cfg.out_channels, model_cfg.image_size, model_cfg.image_size),
            device=device,
            dtype=torch.float32,
            return_trace=bool(cfg["sample"].get("save_trace", False)),
            cond=sample_cond,
            null_cond=null_cond,
            guidance_scale=float(
                args.guidance_scale if args.guidance_scale is not None else cfg["sample"].get("guidance_scale", 1.0)
            ),
            sampler=sampler,
            posterior_noise_scale=(
                args.posterior_noise_scale
                if args.posterior_noise_scale is not None
                else cfg["sample"].get("posterior_noise_scale", None)
            ),
        )

    if is_main_process():
        out = args.out or os.path.join(cfg["logging"]["out_dir"], "sample.png")
        save_sample_grid(result["x"].detach().cpu(), out)
        if "trace" in result:
            torch.save(result["trace"], out.replace(".png", "_trace.pt"))


if __name__ == "__main__":
    main()
