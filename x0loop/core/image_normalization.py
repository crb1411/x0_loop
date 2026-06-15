from __future__ import annotations

from dataclasses import dataclass

import torch


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class ImageNormalization:
    mode: str
    mean: tuple[float, ...]
    std: tuple[float, ...]


def resolve_image_normalization(cfg: dict) -> ImageNormalization:
    ds_cfg = cfg.get("dataset", {}) or {}
    mode = str(ds_cfg.get("normalization", ds_cfg.get("normalize", "minus_one_one"))).lower()
    if mode in {"default", "minus_one_one", "neg_one_to_one", "-1_1"}:
        return ImageNormalization(mode="minus_one_one", mean=(), std=())
    if mode in {"none", "zero_one", "0_1"}:
        return ImageNormalization(mode="zero_one", mean=(), std=())
    if mode in {"standard", "standardize", "mean_std", "cifar10_standard"}:
        dataset_name = str(ds_cfg.get("name", "")).lower()
        default_mean = CIFAR10_MEAN if dataset_name == "cifar10" else (0.5, 0.5, 0.5)
        default_std = CIFAR10_STD if dataset_name == "cifar10" else (0.5, 0.5, 0.5)
        mean = tuple(float(v) for v in ds_cfg.get("mean", default_mean))
        std = tuple(float(v) for v in ds_cfg.get("std", ds_cfg.get("scale", default_std)))
        if len(mean) != len(std):
            raise ValueError(f"dataset mean/std length mismatch: mean={mean}, std={std}")
        if any(v <= 0.0 for v in std):
            raise ValueError(f"dataset std values must be > 0, got {std}")
        return ImageNormalization(mode="standard", mean=mean, std=std)
    raise ValueError(f"Unknown dataset.normalization={mode!r}")


def image_to_display_minus_one_one(x: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """Convert model-space images to the legacy [-1, 1] display space."""
    if cfg is None:
        return x
    norm = resolve_image_normalization(cfg)
    if norm.mode == "minus_one_one":
        return x
    if norm.mode == "zero_one":
        return x * 2.0 - 1.0
    if norm.mode == "standard":
        mean = torch.as_tensor(norm.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.as_tensor(norm.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        if x.ndim == 3:
            mean = mean[0]
            std = std[0]
        x01 = x * std + mean
        return x01 * 2.0 - 1.0
    raise AssertionError(f"Unexpected normalization mode={norm.mode!r}")
