from __future__ import annotations

from dataclasses import dataclass

import torch


class TimeSchedule:
    """Unified schedule for diffusion-like and flow-like processes."""

    def __init__(self, mode: str, num_steps: int, beta_min: float = 0.1, beta_max: float = 20.0):
        self.mode = mode
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

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
            # Standard VP-SDE linear β schedule (DDPM / Song et al. 2020).
            # β(t) = β_min + (β_max − β_min)·t,  α²(t) = exp(−∫₀ᵗ β(s)ds)
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()
            a2 = torch.exp(-integral)
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
