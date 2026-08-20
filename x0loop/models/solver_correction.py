from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SolverCorrectionConfig:
    enabled: bool = False
    solver_steps: int = 20
    start_index: int = 16
    hidden_channels: int = 32
    output_scale: float = 1.0


def build_solver_correction_config(cfg: dict | None) -> SolverCorrectionConfig:
    raw = cfg or {}
    result = SolverCorrectionConfig(
        enabled=bool(raw.get("enabled", False)),
        solver_steps=int(raw.get("solver_steps", 20)),
        start_index=int(raw.get("start_index", 16)),
        hidden_channels=int(raw.get("hidden_channels", 32)),
        output_scale=float(raw.get("output_scale", 1.0)),
    )
    if result.solver_steps <= 0:
        raise ValueError(f"solver_correction.solver_steps must be positive, got {result.solver_steps}")
    if not 0 <= result.start_index < result.solver_steps:
        raise ValueError(
            "solver_correction.start_index must select a solver interval, "
            f"got {result.start_index} for {result.solver_steps} steps"
        )
    if result.hidden_channels <= 0:
        raise ValueError(
            f"solver_correction.hidden_channels must be positive, got {result.hidden_channels}"
        )
    if not math.isfinite(result.output_scale):
        raise ValueError(
            f"solver_correction.output_scale must be finite, got {result.output_scale}"
        )
    return result


class SolverIndexCorrection(nn.Module):
    """Small zero-initialized residual head active on declared solver indices."""

    def __init__(self, cfg: SolverCorrectionConfig, *, channels: int):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_channels
        groups = min(8, hidden)
        while hidden % groups != 0:
            groups -= 1
        self.input = nn.Conv2d(2 * channels, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, hidden)
        self.solver_embedding = nn.Embedding(cfg.solver_steps, hidden)
        self.hidden = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, hidden)
        self.output = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def solver_index(self, t: torch.Tensor) -> torch.Tensor:
        index = torch.round((1.0 - t.float()) * float(self.cfg.solver_steps)).long()
        return index.clamp(0, self.cfg.solver_steps - 1)

    def active_mask(self, t: torch.Tensor) -> torch.Tensor:
        return self.solver_index(t) >= self.cfg.start_index

    def forward(self, x: torch.Tensor, t: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        if t.shape != (x.shape[0],):
            raise ValueError(f"solver correction requires t shape [B], got {tuple(t.shape)}")
        index = self.solver_index(t)
        gate = (index >= self.cfg.start_index).to(dtype=x.dtype).view(-1, 1, 1, 1)
        features = self.input(torch.cat((x, base_output.detach()), dim=1))
        embedding = self.solver_embedding(index).to(dtype=features.dtype).view(features.shape[0], -1, 1, 1)
        features = F.silu(self.norm1(features) + embedding)
        features = F.silu(self.norm2(self.hidden(features)))
        return gate * self.cfg.output_scale * self.output(features).to(dtype=x.dtype)
