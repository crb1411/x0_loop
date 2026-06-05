from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from x0loop.models.embeddings import TimeEmbedMLP


@dataclass
class X0DiscriminatorConfig:
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 16
    num_classes: int = 0
    time_embed_dim: int = 128
    spectral_norm: bool = True
    time_projection: bool = True
    class_projection: bool = True


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


class ResDiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, use_sn: bool):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = _maybe_sn(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), use_sn)
        self.conv2 = _maybe_sn(nn.Conv2d(out_ch, out_ch, 3, padding=1), use_sn)
        if stride != 1 or in_ch != out_ch:
            self.skip = _maybe_sn(nn.Conv2d(in_ch, out_ch, 1, stride=stride), use_sn)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        return (h + self.skip(x)) / 2**0.5


class X0Discriminator(nn.Module):
    """Lightweight conditional discriminator for natural x0 vs predicted x0_hat."""

    def __init__(self, cfg: X0DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        c = int(cfg.base_channels)
        use_sn = bool(cfg.spectral_norm)
        self.stem = _maybe_sn(nn.Conv2d(cfg.in_channels, c, 3, padding=1), use_sn)
        self.blocks = nn.Sequential(
            ResDiscBlock(c, c * 2, stride=2, use_sn=use_sn),
            ResDiscBlock(c * 2, c * 4, stride=2, use_sn=use_sn),
            ResDiscBlock(c * 4, c * 8, stride=2, use_sn=use_sn),
            ResDiscBlock(c * 8, c * 8, stride=1, use_sn=use_sn),
        )
        feat_dim = c * 8
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.head = _maybe_sn(nn.Linear(feat_dim, 1), use_sn)
        self.time_embed = TimeEmbedMLP(cfg.time_embed_dim) if cfg.time_projection else None
        self.time_proj = _maybe_sn(nn.Linear(cfg.time_embed_dim, feat_dim), use_sn) if cfg.time_projection else None
        if cfg.class_projection and cfg.num_classes > 0:
            self.class_embed = _maybe_sn(nn.Embedding(cfg.num_classes + 1, feat_dim), use_sn)
        else:
            self.class_embed = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.act(h).sum(dim=(2, 3))
        logit = self.head(h).squeeze(1)
        if self.time_embed is not None and self.time_proj is not None:
            t_feat = self.time_proj(self.time_embed(t.float()))
            logit = logit + (h * t_feat).sum(dim=1) / (h.shape[1] ** 0.5)
        if self.class_embed is not None and cond is not None:
            y_feat = self.class_embed(cond.long().clamp(0, self.cfg.num_classes))
            logit = logit + (h * y_feat).sum(dim=1) / (h.shape[1] ** 0.5)
        return logit


def build_x0_discriminator(cfg: dict) -> X0Discriminator:
    dc = cfg.get("discriminator", {}) or {}
    mc = cfg.get("model", {}) or {}
    disc_cfg = X0DiscriminatorConfig(
        image_size=int(dc.get("image_size", mc.get("image_size", 32))),
        in_channels=int(dc.get("in_channels", mc.get("in_channels", 3))),
        base_channels=int(dc.get("base_channels", 16)),
        num_classes=int(dc.get("num_classes", mc.get("num_classes", 0))),
        time_embed_dim=int(dc.get("time_embed_dim", 128)),
        spectral_norm=bool(dc.get("spectral_norm", True)),
        time_projection=bool(dc.get("time_projection", True)),
        class_projection=bool(dc.get("class_projection", True)),
    )
    return X0Discriminator(disc_cfg)
