from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from x0loop.models.embeddings import TimeEmbedMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# 2D RoPE helpers
# ---------------------------------------------------------------------------

def _build_2d_rope_cache(h: int, w: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute 2D RoPE cos/sin tables of shape [h*w, head_dim].

    head_dim is split into quarters: first quarter pairs → y-axis frequencies,
    second quarter pairs → x-axis frequencies.  Requires head_dim % 4 == 0.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    quarter = head_dim // 4
    theta = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))  # [quarter]

    y_pos = torch.arange(h, dtype=torch.float32)
    x_pos = torch.arange(w, dtype=torch.float32)

    y_freqs = torch.outer(y_pos, theta).unsqueeze(1).expand(h, w, quarter)  # [h, w, quarter]
    x_freqs = torch.outer(x_pos, theta).unsqueeze(0).expand(h, w, quarter)  # [h, w, quarter]

    # [N, head_dim//2]: y freqs then x freqs, row-major over the grid
    freqs = torch.cat([y_freqs, x_freqs], dim=-1).reshape(h * w, head_dim // 2)

    # Each scalar freq is shared by its (even, odd) pair → repeat_interleave → [N, head_dim]
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs (2i, 2i+1) by 90°: (a, b) → (−b, a)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x [B, heads, N, head_dim] using cos/sin [N, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def build_norm(norm_layer: str, hidden_size: int, eps: float, elementwise_affine: bool):
    name = norm_layer.lower()
    if name in {"layernorm", "ln"}:
        return nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
    if name in {"rmsnorm", "rms"}:
        return nn.RMSNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
    raise ValueError(f"Unsupported norm_layer: {norm_layer}. Use 'layernorm' or 'rmsnorm'.")


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_grid: tuple[int, int] | None = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if rope_grid is not None:
            h, w = rope_grid
            cos, sin = _build_2d_rope_cache(h, w, self.head_dim, base=rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)  # [N, head_dim]
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, heads, N, head_dim]

        if self.rope_cos is not None:
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        norm_layer: str = "layernorm",
        norm_eps: float = 1e-6,
        rope_grid: tuple[int, int] | None = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = build_norm(norm_layer, hidden_size, eps=norm_eps, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=dropout, proj_drop=dropout,
                              rope_grid=rope_grid, rope_base=rope_base)
        self.norm2 = build_norm(norm_layer, hidden_size, eps=norm_eps, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        norm_layer: str = "layernorm",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm_final = build_norm(norm_layer, hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@dataclass
class DiTConfig:
    image_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    qkv_bias: bool = True
    num_classes: int = 0
    cond_dim: int = 0
    norm_layer: str = "layernorm"
    norm_eps: float = 1e-6
    rope_base: float = 10000.0


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.image_size % cfg.patch_size == 0, "image_size must be divisible by patch_size"

        self.patch = nn.Conv2d(cfg.in_channels, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.h_tokens = cfg.image_size // cfg.patch_size
        self.w_tokens = cfg.image_size // cfg.patch_size
        self.num_tokens = self.h_tokens * self.w_tokens

        self.time_mlp = TimeEmbedMLP(dim=cfg.dim)
        self.null_class_id = cfg.num_classes if cfg.num_classes > 0 else None
        self.label_emb = nn.Embedding(cfg.num_classes + 1, cfg.dim) if cfg.num_classes > 0 else None
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.dim) if cfg.cond_dim > 0 else None
        rope_grid = (self.h_tokens, self.w_tokens)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=cfg.dim,
                    num_heads=cfg.heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    qkv_bias=cfg.qkv_bias,
                    norm_layer=cfg.norm_layer,
                    norm_eps=cfg.norm_eps,
                    rope_grid=rope_grid,
                    rope_base=cfg.rope_base,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(
            cfg.dim,
            cfg.patch_size,
            cfg.out_channels,
            norm_layer=cfg.norm_layer,
            norm_eps=cfg.norm_eps,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init)

        # adaLN-Zero: start modulation/gates from zero.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _cond_embedding(self, cond: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if cond is None:
            if self.label_emb is not None and self.null_class_id is not None:
                null_cond = torch.full((batch,), self.null_class_id, device=device, dtype=torch.long)
                return self.label_emb(null_cond).to(dtype)
            return torch.zeros(batch, self.cfg.dim, device=device, dtype=dtype)

        if cond.ndim == 1 and cond.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
            if self.label_emb is None:
                raise ValueError("Received class-label cond but DiTConfig.num_classes <= 0.")
            cond = cond.to(device=device, dtype=torch.long)
            if bool((cond < 0).any()) or bool((cond > self.null_class_id).any()):
                raise ValueError(
                    f"Class-label cond must be in [0, {self.null_class_id}], where {self.null_class_id} is the null CFG label."
                )
            return self.label_emb(cond).to(dtype)

        if cond.ndim == 2:
            cond = cond.to(device=device, dtype=dtype)
            if self.cond_proj is not None:
                return self.cond_proj(cond)
            if cond.shape[1] == self.cfg.dim:
                return cond
            raise ValueError("Received vector cond but cond_dim is not configured and cond width != model dim.")

        raise ValueError(f"Unsupported cond shape/dtype: shape={tuple(cond.shape)}, dtype={cond.dtype}")

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.out_channels
        h = self.h_tokens
        w = self.w_tokens
        x = tokens
        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h * p, w * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        tok = self._patchify(x)  # [B, N, D]
        c = self.time_mlp(t)     # [B, D]
        c = c + self._cond_embedding(cond=cond, batch=x.shape[0], device=x.device, dtype=c.dtype)
        for blk in self.blocks:
            tok = blk(tok, c)
        tok = self.final_layer(tok, c)
        return self._unpatchify(tok)
