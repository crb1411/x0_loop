from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TimeScheduleConfig:
    mode: str = "diffusion"
    num_steps: int = 1000
    diffusion_lambda: float = 12.0


class TimeSchedule:
    """Unified schedule for diffusion-like and flow-like processes."""

    def __init__(self, mode: str, num_steps: int, diffusion_lambda: float = 12.0):
        self.mode = mode
        self.num_steps = num_steps
        self.diffusion_lambda = diffusion_lambda

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.mode == "diffusion":
            t_idx = torch.randint(1, self.num_steps + 1, (batch_size,), device=device)
            t = t_idx.float() / float(self.num_steps)
            return t.clamp(1e-5, 1.0)
        if self.mode == "flow":
            return torch.rand(batch_size, device=device).clamp(1e-5, 1.0)
        raise ValueError(f"Unknown schedule mode: {self.mode}")

    def iter_pairs(self, steps: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(steps):
            t = t_grid[i].clamp(1e-5, 1.0)
            s = t_grid[i + 1].clamp(0.0, 1.0)
            out.append((t, s))
        return out

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if self.mode == "diffusion":
            # VP-style closed-form schedule: alpha^2 + sigma^2 = 1.
            a2 = torch.exp(-self.diffusion_lambda * t)
            return a2.sqrt().clamp_min(1e-5)
        if self.mode == "flow":
            return (1.0 - t).clamp_min(1e-5)
        raise ValueError(f"Unknown schedule mode: {self.mode}")

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if self.mode == "diffusion":
            a = self.alpha(t)
            s2 = (1.0 - a.square()).clamp_min(1e-10)
            return s2.sqrt().clamp_min(1e-5)
        if self.mode == "flow":
            return t.clamp_min(1e-5)
        raise ValueError(f"Unknown schedule mode: {self.mode}")

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        s = self.sigma(t)
        return (a.square() / s.square()).clamp_min(1e-8)
