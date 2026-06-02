from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import MNIST

from x0loop.core.config import dump_resolved_config, load_merged_config
from x0loop.models.factory import build_model
from x0loop.train import build_process, build_schedule
from x0loop.utils.checkpoint import load_checkpoint
from x0loop.utils.ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Run runtime yaml (full merged config).")
    p.add_argument("--runtime-config", type=str, default="", help="Optional extra runtime override.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--out-dir-base", type=str, default="")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--real-dir", type=str, default="", help="Optional shared directory for exported real MNIST RGB.")
    p.add_argument("--feature-extractor", type=str, default="inception-v3-compat")
    p.add_argument("--feature-layer-fid", type=str, default="2048")
    p.add_argument("--cache-root", type=str, default="", help="torch-fidelity cache root.")
    p.add_argument("--cache-name-prefix", type=str, default="mnist")
    p.add_argument("--sampler", type=str, default="auto", choices=["auto", "ddim", "posterior", "euler", "heun"])
    p.add_argument("--posterior-noise-scale", type=float, default=None)
    return p.parse_args()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_eval_output_dir(args) -> str:
    if args.out_dir:
        return args.out_dir
    if not args.out_dir_base:
        raise ValueError("Either --out-dir or --out-dir-base must be set.")

    timestamp = os.environ.get("X0LOOP_RUN_TIMESTAMP")
    if not timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        os.environ["X0LOOP_RUN_TIMESTAMP"] = timestamp
    return str(Path(args.out_dir_base) / f"{timestamp}_eval")


def setup_file_logging(out_dir: str) -> str:
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    fp = log_file.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(orig_stdout, fp)
    sys.stderr = _Tee(orig_stderr, fp)

    def cleanup():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        fp.flush()
        fp.close()

    atexit.register(cleanup)
    return str(log_file)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    x = ((x.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c >= 3:
        x = x[:3]
    else:
        pad = torch.zeros((3 - c, h, w), dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    return x.permute(1, 2, 0).contiguous().numpy()


def export_real_mnist(root: str, out_dir: str, split: str):
    os.makedirs(out_dir, exist_ok=True)
    train = split == "train"
    ds = MNIST(root=root, train=train, download=False, transform=None)
    for i in range(len(ds)):
        img, _ = ds[i]  # PIL grayscale
        img = img.convert("RGB")
        img.save(os.path.join(out_dir, f"{i:06d}.png"))


@torch.no_grad()
def export_fake_images(
    *,
    model,
    process,
    out_dir: str,
    num_samples: int,
    batch_size: int,
    steps: int,
    image_size: int,
    out_channels: int,
    device: torch.device,
    sampler: str | None,
    posterior_noise_scale: float | None,
):
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    while done < num_samples:
        bs = min(batch_size, num_samples - done)
        result = process.sample(
            model=model,
            steps=steps,
            shape=(bs, out_channels, image_size, image_size),
            device=device,
            dtype=torch.float32,
            return_trace=False,
            cond=None,
            sampler=sampler,
            posterior_noise_scale=posterior_noise_scale,
        )
        x = result["x"]
        for i in range(bs):
            arr = tensor_to_rgb_uint8(x[i])
            Image.fromarray(arr, mode="RGB").save(os.path.join(out_dir, f"{done + i:06d}.png"))
        done += bs


def main():
    args = parse_args()
    out_dir = resolve_eval_output_dir(args)
    log_file = setup_file_logging(out_dir)
    print(f"[fid] output_dir={out_dir}", flush=True)
    print(f"[fid] log_file={log_file}", flush=True)

    cfg = load_merged_config(args.config, args.runtime_config if args.runtime_config else None, resolve_logging=False)
    cfg.setdefault("logging", {})
    cfg["logging"]["out_dir"] = out_dir
    resolved_config_path = dump_resolved_config(cfg, out_dir)
    print(f"[fid] resolved_config={resolved_config_path}", flush=True)

    ds_name = str(cfg.get("dataset", {}).get("name", "")).lower()
    if ds_name != "mnist":
        raise ValueError(f"This script is MNIST-only, but dataset.name={ds_name}")

    set_seed(args.seed)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = build_model(cfg["model"])
    model = model.to(device)

    use_ema = bool(cfg.get("train", {}).get("use_ema", True) and args.use_ema)
    ema = EMA(model, decay=float(cfg.get("train", {}).get("ema_decay", 0.9999))) if use_ema else None
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
        info = ckpt.get("_load_info", {})
        print(
            "[fid][ckpt] fallback strict=False used; "
            f"missing={len(info.get('missing_keys', []))}, unexpected={len(info.get('unexpected_keys', []))}",
            flush=True,
        )

    if ema is not None:
        ema.copy_to(model)

    model.eval()
    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule)

    steps = int(args.steps if args.steps > 0 else cfg.get("sample", {}).get("steps", 100))
    fake_dir = os.path.join(out_dir, "fake")
    if args.real_dir:
        real_dir = args.real_dir
    else:
        real_dir = os.path.join(cfg["dataset"]["root"], f"fid_real_rgb_{args.real_split}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(real_dir) or len(os.listdir(real_dir)) == 0:
        print(f"[fid] exporting real MNIST ({args.real_split}) -> {real_dir}", flush=True)
        export_real_mnist(root=cfg["dataset"]["root"], out_dir=real_dir, split=args.real_split)

    print(f"[fid] exporting fake images -> {fake_dir}", flush=True)
    sampler = None if args.sampler == "auto" else args.sampler
    if sampler is None:
        sampler = cfg.get("sample", {}).get("sampler", None)
        if isinstance(sampler, str) and sampler.lower() == "auto":
            sampler = None
    posterior_noise_scale = (
        args.posterior_noise_scale
        if args.posterior_noise_scale is not None
        else cfg.get("sample", {}).get("posterior_noise_scale", None)
    )
    if sampler is not None:
        print(f"[fid] sampler={sampler}, posterior_noise_scale={posterior_noise_scale}", flush=True)
    export_fake_images(
        model=model,
        process=process,
        out_dir=fake_dir,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        steps=steps,
        image_size=int(model_cfg.image_size),
        out_channels=int(model_cfg.out_channels),
        device=device,
        sampler=sampler,
        posterior_noise_scale=posterior_noise_scale,
    )

    from torch_fidelity import calculate_metrics

    print(
        "[fid] NOTE: Standard FID uses a fixed feature extractor "
        f"({args.feature_extractor}, layer={args.feature_layer_fid}), not the DiT model feature space.",
        flush=True,
    )
    print("[fid] calculating...", flush=True)
    metric_kwargs = dict(
        input1=fake_dir,
        input2=real_dir,
        cuda=bool(device.type == "cuda"),
        isc=False,
        fid=True,
        kid=False,
        verbose=True,
        cache=True,
        feature_extractor=args.feature_extractor,
        feature_layer_fid=args.feature_layer_fid,
        input2_cache_name=f"{args.cache_name_prefix}_{args.real_split}_{args.feature_extractor}_{args.feature_layer_fid}",
    )
    if args.cache_root:
        metric_kwargs["cache_root"] = args.cache_root

    metrics = calculate_metrics(**metric_kwargs)

    result = {
        "ckpt": args.ckpt,
        "config": args.config,
        "num_samples": int(args.num_samples),
        "steps": int(steps),
        "real_split": args.real_split,
        "real_dir": real_dir,
        "feature_extractor": args.feature_extractor,
        "feature_layer_fid": args.feature_layer_fid,
        "sampler": sampler if sampler is not None else "process_default",
        "posterior_noise_scale": posterior_noise_scale,
        "metrics": metrics,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    out_json = os.path.join(out_dir, "fid_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)
    print(f"[fid] result saved: {out_json}", flush=True)
    print(f"[fid] metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()
