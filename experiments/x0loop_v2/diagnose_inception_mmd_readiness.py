"""Readiness gate for a fixed, differentiable Inception KID objective.

The generator remains frozen. The diagnostic checks that the exact FID feature
network separates terminal and real distributions, that stochastic real
references induce consistent pixel gradients, and that a tiny projected pixel
step improves an independent-reference objective.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

from x0loop.core.image_normalization import image_to_display_minus_one_one
from x0loop.losses.inception_mmd import (
    DifferentiableFIDInception,
    model_images_to_fid_pixels,
    polynomial_mmd2,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a frozen Inception-MMD distribution objective.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed-dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-trials", type=int, default=100)
    parser.add_argument("--gradient-batch-size", type=int, default=16)
    parser.add_argument("--gradient-references", type=int, default=4)
    parser.add_argument("--descent-trials", type=int, default=4)
    parser.add_argument("--step-rms", type=float, default=0.01)
    parser.add_argument("--separation-z", type=float, default=3.0)
    parser.add_argument("--gradient-cosine", type=float, default=0.20)
    parser.add_argument("--descent-pass-fraction", type=float, default=0.75)
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


def _make_extractor(device: torch.device) -> FeatureExtractorInceptionV3:
    extractor = FeatureExtractorInceptionV3(
        "inception-v3-compat",
        ["2048"],
        feature_extractor_internal_dtype="float32",
        verbose=False,
    ).to(device)
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


@torch.inference_mode()
def _extract(
    extractor: FeatureExtractorInceptionV3,
    images: torch.Tensor,
    cfg: dict,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    result: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device).float()
        pixels = model_images_to_fid_pixels(batch, cfg, straight_through_quantize=False)
        result.append(extractor(pixels)[0].cpu())
    return torch.cat(result)


def _load_or_extract_features(
    *,
    extractor: FeatureExtractorInceptionV3,
    fixed: dict,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    path: Path,
) -> tuple[dict[str, torch.Tensor], float]:
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        return cached, 0.0
    started = time.perf_counter()
    result = {
        "real": _extract(extractor, fixed["real"], cfg, device=device, batch_size=batch_size),
        "fake": _extract(extractor, fixed["fake"], cfg, device=device, batch_size=batch_size),
    }
    duration = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result, duration


def _sample_indices(count: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    if batch_size > count:
        raise ValueError(f"batch size {batch_size} exceeds available feature count {count}")
    return torch.randperm(count, generator=generator)[:batch_size]


def _bootstrap_separation(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    batch_size: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    fake_real: list[float] = []
    real_real: list[float] = []
    for _ in range(trials):
        fake_index = _sample_indices(fake.shape[0], batch_size, generator)
        real_a_index = _sample_indices(real.shape[0], batch_size, generator)
        real_b_index = _sample_indices(real.shape[0], batch_size, generator)
        fake_real.append(float(polynomial_mmd2(fake[fake_index], real[real_a_index])))
        real_real.append(float(polynomial_mmd2(real[real_a_index], real[real_b_index])))
    fake_mean = statistics.fmean(fake_real)
    real_mean = statistics.fmean(real_real)
    fake_std = statistics.stdev(fake_real)
    real_std = statistics.stdev(real_real)
    standard_error = math.sqrt((fake_std * fake_std + real_std * real_std) / float(trials))
    return {
        "fake_real_mean": fake_mean,
        "fake_real_std": fake_std,
        "real_real_mean": real_mean,
        "real_real_std": real_std,
        "mean_gap": fake_mean - real_mean,
        "gap_z": (fake_mean - real_mean) / max(standard_error, 1.0e-12),
        "trials": trials,
        "batch_size": batch_size,
    }


def _pairwise_cosines(vectors: list[torch.Tensor]) -> list[float]:
    result: list[float] = []
    for i, left in enumerate(vectors):
        for right in vectors[i + 1 :]:
            result.append(float(torch.nn.functional.cosine_similarity(left, right, dim=0)))
    return result


def _gradient_consistency(
    *,
    differentiable: DifferentiableFIDInception,
    fake_images: torch.Tensor,
    real_features: torch.Tensor,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    references: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    fake_index = _sample_indices(fake_images.shape[0], batch_size, generator)
    images = fake_images[fake_index].to(device).float().requires_grad_(True)
    pixels = model_images_to_fid_pixels(images, cfg, straight_through_quantize=True)
    fake_feature = differentiable(pixels)
    gradients: list[torch.Tensor] = []
    losses: list[float] = []
    for reference_index in range(references):
        real_index = _sample_indices(real_features.shape[0], batch_size, generator)
        real = real_features[real_index].to(device)
        loss = polynomial_mmd2(fake_feature, real)
        gradient = torch.autograd.grad(
            loss,
            images,
            retain_graph=reference_index + 1 < references,
        )[0]
        gradients.append(gradient.detach().flatten().float().cpu())
        losses.append(float(loss.detach()))
    cosines = _pairwise_cosines(gradients)
    return {
        "losses": losses,
        "pairwise_cosines": cosines,
        "median_cosine": statistics.median(cosines),
        "min_cosine": min(cosines),
        "gradient_rms": [float(vector.square().mean().sqrt()) for vector in gradients],
        "batch_size": batch_size,
        "references": references,
    }


def _projected_descent(
    *,
    differentiable: DifferentiableFIDInception,
    fake_images: torch.Tensor,
    real_train: torch.Tensor,
    real_val: torch.Tensor,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    trials: int,
    step_rms: float,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, float | bool]] = []
    for _ in range(trials):
        fake_index = _sample_indices(fake_images.shape[0], batch_size, generator)
        train_index = _sample_indices(real_train.shape[0], batch_size, generator)
        val_index = _sample_indices(real_val.shape[0], batch_size, generator)
        images = fake_images[fake_index].to(device).float().requires_grad_(True)
        before_feature = differentiable(
            model_images_to_fid_pixels(images, cfg, straight_through_quantize=True)
        )
        train_loss = polynomial_mmd2(before_feature, real_train[train_index].to(device))
        gradient = torch.autograd.grad(train_loss, images)[0]
        gradient_rms = gradient.float().square().mean().sqrt().clamp_min(1.0e-12)
        proposal = (images - float(step_rms) * gradient / gradient_rms).detach()
        after_feature = differentiable(
            model_images_to_fid_pixels(proposal, cfg, straight_through_quantize=True)
        )
        val_reference = real_val[val_index].to(device)
        before_val = polynomial_mmd2(before_feature.detach(), val_reference)
        after_val = polynomial_mmd2(after_feature, val_reference)
        before_display = image_to_display_minus_one_one(images.detach(), cfg=cfg)
        after_display = image_to_display_minus_one_one(proposal, cfg=cfg)
        rows.append({
            "train_loss": float(train_loss.detach()),
            "before_val": float(before_val.detach()),
            "after_val": float(after_val.detach()),
            "delta_val": float((after_val - before_val).detach()),
            "improved": bool(after_val < before_val),
            "model_step_rms": float((proposal - images.detach()).float().square().mean().sqrt()),
            "display_step_rms": float((after_display - before_display).float().square().mean().sqrt()),
        })
    improved = sum(int(row["improved"]) for row in rows)
    return {
        "rows": rows,
        "improved": improved,
        "trials": trials,
        "improved_fraction": improved / float(trials),
        "median_delta_val": statistics.median(float(row["delta_val"]) for row in rows),
        "step_rms": step_rms,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    separation = result["separation"]
    gradients = result["gradient_consistency"]
    descent = result["projected_descent"]
    decision = result["decision"]
    lines = [
        "# Cycle 06 Inception-MMD readiness",
        "",
        f"Frozen terminal distribution: `{result['checkpoint']}`; exact FID Inception 2048 features.",
        "",
        "| check | value | threshold | pass |",
        "|---|---:|---:|---:|",
        f"| forward max abs error | {result['forward_equivalence_max_abs']:.6g} | 1e-5 | {decision['forward_equivalence']} |",
        f"| fake-real vs real-real gap z | {separation['gap_z']:.4f} | {decision['thresholds']['separation_z']:.2f} | {decision['separation']} |",
        f"| reference-gradient median cosine | {gradients['median_cosine']:.4f} | {decision['thresholds']['gradient_cosine']:.2f} | {decision['gradient_consistency']} |",
        f"| independent-reference descent | {descent['improved']}/{descent['trials']} | {decision['thresholds']['descent_pass_fraction']:.2f} | {decision['projected_descent']} |",
        "",
        f"Overall readiness pass: **{decision['pass']}**.",
        "",
        "This diagnostic freezes the generator; it is not a training or FID result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.fixed_dataset)
    out = Path(args.out)
    if not checkpoint_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"checkpoint={checkpoint_path}, fixed_dataset={dataset_path}")
    if args.gradient_references < 2 or args.descent_trials <= 0:
        raise ValueError("gradient-references must be >=2 and descent-trials must be positive")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config") or {}
    fixed = torch.load(dataset_path, map_location="cpu", weights_only=False)
    n_train = int(fixed["num_train"])
    n_val = int(fixed["num_val"])
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    extractor = _make_extractor(device)
    features, extract_s = _load_or_extract_features(
        extractor=extractor,
        fixed=fixed,
        cfg=cfg,
        device=device,
        batch_size=args.extract_batch_size,
        path=out / "fixed_inception_features.pt",
    )
    real_train, real_val = features["real"][:n_train], features["real"][n_train : n_train + n_val]
    fake_train, fake_val = features["fake"][:n_train], features["fake"][n_train : n_train + n_val]
    differentiable = DifferentiableFIDInception(extractor).to(device).eval()

    check_images = fixed["fake"][:4].to(device).float()
    check_features = differentiable(
        model_images_to_fid_pixels(check_images, cfg, straight_through_quantize=True)
    ).detach().cpu()
    forward_error = float((check_features - features["fake"][:4]).abs().max())
    separation = _bootstrap_separation(
        real_val,
        fake_val,
        batch_size=args.bootstrap_batch_size,
        trials=args.bootstrap_trials,
        seed=args.seed + 101,
    )
    gradients = _gradient_consistency(
        differentiable=differentiable,
        fake_images=fixed["fake"][:n_train],
        real_features=real_train,
        cfg=cfg,
        device=device,
        batch_size=args.gradient_batch_size,
        references=args.gradient_references,
        seed=args.seed + 211,
    )
    descent = _projected_descent(
        differentiable=differentiable,
        fake_images=fixed["fake"][:n_train],
        real_train=real_train,
        real_val=real_val,
        cfg=cfg,
        device=device,
        batch_size=args.gradient_batch_size,
        trials=args.descent_trials,
        step_rms=args.step_rms,
        seed=args.seed + 307,
    )
    thresholds = {
        "separation_z": args.separation_z,
        "gradient_cosine": args.gradient_cosine,
        "descent_pass_fraction": args.descent_pass_fraction,
    }
    checks = {
        "forward_equivalence": forward_error <= 1.0e-5,
        "separation": separation["mean_gap"] > 0.0 and separation["gap_z"] >= args.separation_z,
        "gradient_consistency": gradients["median_cosine"] >= args.gradient_cosine,
        "projected_descent": descent["improved_fraction"] >= args.descent_pass_fraction,
    }
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "fixed_dataset": str(dataset_path),
        "seed": args.seed,
        "num_train": n_train,
        "num_val": n_val,
        "feature_extract_s": extract_s,
        "forward_equivalence_max_abs": forward_error,
        "separation": separation,
        "gradient_consistency": gradients,
        "projected_descent": descent,
        "decision": {**checks, "thresholds": thresholds, "pass": all(checks.values())},
    }
    (out / "inception_mmd_readiness.json").write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(result, out / "inception_mmd_readiness.md")
    print(json.dumps(_json_ready(result["decision"]), sort_keys=True), flush=True)
    print(f"wrote {out / 'inception_mmd_readiness.json'}", flush=True)


if __name__ == "__main__":
    main()
