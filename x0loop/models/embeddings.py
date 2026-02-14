from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding, t expected in [0, 1] float."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.SiLU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, dim=self.net[0].in_features)
        return self.net(emb)
