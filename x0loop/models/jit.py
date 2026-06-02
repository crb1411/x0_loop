from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from x0loop.models.dit import _apply_rope, _build_2d_rope_cache, modulate
from x0loop.models.embeddings import TimeEmbedMLP


def _build_2d_sincos_position_embedding(grid_size: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("JiT dim must be divisible by 4 for 2D positional embeddings.")
    quarter = dim // 4
    omega = torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter)
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    y = y.flatten().float().unsqueeze(1) * omega.unsqueeze(0)
    x = x.flatten().float().unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], dim=1).unsqueeze(0)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, bottleneck_dim: int, dim: int):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} input, got {tuple(x.shape[-2:])}.")
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("JiT dim must be divisible by heads.")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        b, n, dim = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        rope_cos = rope_cos.to(dtype=q.dtype)
        rope_sin = rope_sin.to(dtype=q.dtype)
        q = _apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = _apply_rope(self.k_norm(k), rope_cos, rope_sin)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(b, n, dim))


class JiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLUFFN(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn), rope_cos, rope_sin)
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class JiTConfig:
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int = 4
    dim: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_classes: int = 0
    cond_dim: int = 0
    bottleneck_dim: int = 128
    in_context_len: int = 32
    in_context_start: int = 4


class JiT(nn.Module):
    """Just image Transformer adapted to the x0loop backbone interface."""

    def __init__(self, cfg: JiTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("JiT image_size must be divisible by patch_size.")
        if cfg.in_context_len > 0 and not 0 <= cfg.in_context_start < cfg.depth:
            raise ValueError("JiT in_context_start must select a transformer block.")

        self.h_tokens = cfg.image_size // cfg.patch_size
        self.w_tokens = self.h_tokens
        self.num_tokens = self.h_tokens * self.w_tokens
        self.patch = BottleneckPatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.bottleneck_dim, cfg.dim)
        self.time_mlp = TimeEmbedMLP(dim=cfg.dim)
        self.null_class_id = cfg.num_classes if cfg.num_classes > 0 else None
        self.label_emb = nn.Embedding(cfg.num_classes + 1, cfg.dim) if cfg.num_classes > 0 else None
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.dim) if cfg.cond_dim > 0 else None
        self.blocks = nn.ModuleList([JiTBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)])
        self.final_layer = FinalLayer(cfg.dim, cfg.patch_size, cfg.out_channels)

        self.register_buffer("pos_embed", _build_2d_sincos_position_embedding(self.h_tokens, cfg.dim), persistent=False)
        rope_cos, rope_sin = _build_2d_rope_cache(self.h_tokens, self.w_tokens, cfg.dim // cfg.heads)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.context_pos = nn.Parameter(torch.zeros(1, cfg.in_context_len, cfg.dim))
        context_cos = torch.ones(cfg.in_context_len, rope_cos.shape[1])
        context_sin = torch.zeros(cfg.in_context_len, rope_sin.shape[1])
        self.register_buffer("context_rope_cos", torch.cat([context_cos, rope_cos]), persistent=False)
        self.register_buffer("context_rope_sin", torch.cat([context_sin, rope_sin]), persistent=False)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        nn.init.normal_(self.context_pos, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _cond_embedding(self, cond: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if cond is None:
            if self.label_emb is None:
                return torch.zeros(batch, self.cfg.dim, device=device, dtype=dtype)
            cond = torch.full((batch,), self.null_class_id, device=device, dtype=torch.long)
        if cond.ndim == 1:
            if self.label_emb is None:
                raise ValueError("Received class-label cond but JiTConfig.num_classes <= 0.")
            cond = cond.to(device=device, dtype=torch.long)
            if bool((cond < 0).any()) or bool((cond > self.null_class_id).any()):
                raise ValueError(f"JiT class-label cond must be in [0, {self.null_class_id}].")
            return self.label_emb(cond).to(dtype)
        if cond.ndim == 2:
            cond = cond.to(device=device, dtype=dtype)
            if self.cond_proj is not None:
                return self.cond_proj(cond)
            if cond.shape[1] == self.cfg.dim:
                return cond
        raise ValueError(f"Unsupported JiT cond shape: {tuple(cond.shape)}")

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b, _, _ = tokens.shape
        p, c = self.cfg.patch_size, self.cfg.out_channels
        x = tokens.view(b, self.h_tokens, self.w_tokens, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, self.cfg.image_size, self.cfg.image_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        cond_embed = self._cond_embedding(cond, batch=x.shape[0], device=x.device, dtype=x.dtype)
        c = self.time_mlp(t) + cond_embed
        tokens = self.patch(x)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        for index, block in enumerate(self.blocks):
            if self.cfg.in_context_len > 0 and index == self.cfg.in_context_start:
                context = cond_embed.unsqueeze(1) + self.context_pos.to(dtype=cond_embed.dtype)
                tokens = torch.cat([context, tokens], dim=1)
            has_context = self.cfg.in_context_len > 0 and index >= self.cfg.in_context_start
            rope = (self.context_rope_cos, self.context_rope_sin) if has_context else (self.rope_cos, self.rope_sin)
            tokens = block(tokens, c, *rope)
        if self.cfg.in_context_len > 0:
            tokens = tokens[:, self.cfg.in_context_len:]
        return self._unpatchify(self.final_layer(tokens, c))


def _preset(**defaults):
    def build(**kwargs):
        return JiT(JiTConfig(**(defaults | kwargs)))
    return build


JiT_B_16 = _preset(depth=12, dim=768, heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16)
JiT_B_32 = _preset(depth=12, dim=768, heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32)
JiT_L_16 = _preset(depth=24, dim=1024, heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16)
JiT_L_32 = _preset(depth=24, dim=1024, heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32)
JiT_H_16 = _preset(depth=32, dim=1280, heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16)
JiT_H_32 = _preset(depth=32, dim=1280, heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32)

JiT_models = {
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
}


def build_jit(preset: str | None = None, **kwargs) -> JiT:
    if preset is None:
        return JiT(JiTConfig(**kwargs))
    if preset not in JiT_models:
        choices = ", ".join(JiT_models)
        raise ValueError(f"Unknown JiT preset={preset!r}; use one of: {choices}")
    return JiT_models[preset](**kwargs)
