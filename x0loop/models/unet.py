from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from x0loop.models.embeddings import TimeEmbedMLP


def _as_tuple(x) -> tuple[int, ...]:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    return (int(x),)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.norm1 = _group_norm(self.in_ch)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, self.out_ch)
        self.norm2 = _group_norm(self.out_ch)
        self.drop = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if self.in_ch == self.out_ch else nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        if self.channels % self.num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.head_dim = self.channels // self.num_heads
        self.norm = _group_norm(self.channels)
        self.qkv = nn.Conv1d(self.channels, self.channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).view(b, c, h * w)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        attn = torch.softmax((q @ k) * (self.head_dim ** -0.5), dim=-1)
        out = attn @ v
        out = out.transpose(-1, -2).reshape(b, c, h * w)
        out = self.proj(out).view(b, c, h, w)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


@dataclass
class UNetConfig:
    name: str = "unet"
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_mult: tuple[int, ...] | list[int] = field(default_factory=lambda: (1, 2, 2, 4))
    num_res_blocks: int = 3
    dropout: float = 0.0
    num_classes: int = 0
    cond_dim: int = 0
    time_dim_mult: int = 4
    attention_resolutions: tuple[int, ...] | list[int] = field(default_factory=lambda: (16, 8))
    attention_heads: int = 4


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.image_size = int(cfg.image_size)
        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.out_channels)
        self.base_channels = int(cfg.base_channels)
        self.channel_mult = _as_tuple(cfg.channel_mult)
        self.attention_resolutions = set(_as_tuple(cfg.attention_resolutions))
        self.num_res_blocks = int(cfg.num_res_blocks)
        self.time_dim = self.base_channels * int(cfg.time_dim_mult)

        self.time_mlp_in = TimeEmbedMLP(dim=self.base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.null_class_id = cfg.num_classes if cfg.num_classes > 0 else None
        self.label_emb = nn.Embedding(cfg.num_classes + 1, self.time_dim) if cfg.num_classes > 0 else None
        self.cond_proj = nn.Linear(cfg.cond_dim, self.time_dim) if cfg.cond_dim > 0 else None

        self.input_conv = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.downsample_flags: list[bool] = []
        skip_channels: list[int] = [self.base_channels]
        ch = self.base_channels
        resolution = self.image_size
        for level, mult in enumerate(self.channel_mult):
            out_ch = self.base_channels * int(mult)
            for _ in range(self.num_res_blocks):
                layers = nn.ModuleList([ResBlock(ch, out_ch, self.time_dim, dropout=cfg.dropout)])
                ch = out_ch
                if resolution in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=int(cfg.attention_heads)))
                self.downs.append(layers)
                self.downsample_flags.append(False)
                skip_channels.append(ch)
            if level != len(self.channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                self.downsample_flags.append(True)
                resolution //= 2
                skip_channels.append(ch)

        self.mid1 = ResBlock(ch, ch, self.time_dim, dropout=cfg.dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=int(cfg.attention_heads))
        self.mid2 = ResBlock(ch, ch, self.time_dim, dropout=cfg.dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch = self.base_channels * int(mult)
            for _ in range(self.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                layers = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, self.time_dim, dropout=cfg.dropout)])
                ch = out_ch
                if resolution in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=int(cfg.attention_heads)))
                self.ups.append(layers)
            if level != 0:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                resolution *= 2

        self.out_norm = _group_norm(ch)
        self.out_conv = nn.Conv2d(ch, self.out_channels, kernel_size=3, padding=1)
        self.num_tokens = self.image_size * self.image_size
        self.h_tokens = self.image_size
        self.w_tokens = self.image_size
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def _conditioning(self, t: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_mlp(self.time_mlp_in(t))
        if self.label_emb is not None:
            if cond is None:
                cond = torch.full((t.shape[0],), self.null_class_id, device=t.device, dtype=torch.long)
            emb = emb + self.label_emb(cond.long())
        elif self.cond_proj is not None and cond is not None:
            emb = emb + self.cond_proj(cond)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._conditioning(t, cond)
        h = self.input_conv(x)
        skips = [h]

        for layers, is_downsample in zip(self.downs, self.downsample_flags):
            if is_downsample:
                h = layers[0](h)
                skips.append(h)
                continue
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            skips.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        for layers in self.ups:
            if len(layers) == 1 and isinstance(layers[0], Upsample):
                h = layers[0](h)
                continue
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))
