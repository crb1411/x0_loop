from __future__ import annotations

import torch.nn as nn

from x0loop.models.dit import DiT, DiTConfig
from x0loop.models.jit import build_jit
from x0loop.models.unet import UNet, UNetConfig


def build_model(config: dict) -> tuple[nn.Module, object]:
    model_cfg = dict(config)
    name = str(model_cfg.pop("name", "dit")).lower()
    if name == "dit":
        cfg = DiTConfig(**model_cfg)
        return DiT(cfg), cfg
    if name == "unet":
        cfg = UNetConfig(**model_cfg)
        return UNet(cfg), cfg
    if name == "jit":
        model = build_jit(**model_cfg)
        return model, model.cfg
    raise ValueError(f"Unknown model.name={name!r}; use dit | unet | jit")
