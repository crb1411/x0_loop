from __future__ import annotations

import torch


class TimeSampler:
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class LegacyScheduleSampler(TimeSampler):
    def __init__(self, schedule):
        self.schedule = schedule

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.schedule.sample_t(batch_size, device=device)


class UniformContinuousSampler(TimeSampler):
    def __init__(self, *, min_t: float = 1e-5, max_t: float = 1.0):
        self.min_t = float(min_t)
        self.max_t = float(max_t)
        if not (0.0 <= self.min_t < self.max_t <= 1.0):
            raise ValueError(f"Require 0 <= min_t < max_t <= 1, got min_t={min_t}, max_t={max_t}")

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.rand(batch_size, device=device)
        return self.min_t + (self.max_t - self.min_t) * t


class UniformDiscreteSampler(TimeSampler):
    def __init__(self, *, num_steps: int, min_step: int = 1, max_step: int | None = None):
        self.num_steps = int(num_steps)
        self.min_step = int(min_step)
        self.max_step = int(max_step) if max_step is not None else self.num_steps
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if not (1 <= self.min_step <= self.max_step <= self.num_steps):
            raise ValueError(
                f"Require 1 <= min_step <= max_step <= num_steps, got "
                f"min_step={self.min_step}, max_step={self.max_step}, num_steps={self.num_steps}"
            )

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(self.min_step, self.max_step + 1, (batch_size,), device=device)
        return (idx.float() / float(self.num_steps)).clamp(1e-5, 1.0)


class LogitNormalSampler(TimeSampler):
    def __init__(self, *, mean: float = 0.0, std: float = 1.0, min_t: float = 1e-5, max_t: float = 1.0 - 1e-5):
        self.mean = float(mean)
        self.std = float(std)
        self.min_t = float(min_t)
        self.max_t = float(max_t)
        if self.std <= 0.0:
            raise ValueError(f"std must be > 0, got {std}")
        if not (0.0 <= self.min_t < self.max_t <= 1.0):
            raise ValueError(f"Require 0 <= min_t < max_t <= 1, got min_t={min_t}, max_t={max_t}")

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, device=device) * self.std + self.mean
        return torch.sigmoid(z).clamp(self.min_t, self.max_t)


class BetaSampler(TimeSampler):
    def __init__(self, *, alpha: float = 1.0, beta: float = 1.0, min_t: float = 1e-5, max_t: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_t = float(min_t)
        self.max_t = float(max_t)
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError(f"alpha and beta must be > 0, got alpha={alpha}, beta={beta}")
        if not (0.0 <= self.min_t < self.max_t <= 1.0):
            raise ValueError(f"Require 0 <= min_t < max_t <= 1, got min_t={min_t}, max_t={max_t}")

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dist = torch.distributions.Beta(
            torch.tensor(self.alpha, device=device),
            torch.tensor(self.beta, device=device),
        )
        t = dist.sample((batch_size,))
        return self.min_t + (self.max_t - self.min_t) * t


def build_time_sampler(cfg: dict, schedule) -> TimeSampler:
    scfg = cfg.get("time_sampler", None)
    if scfg is None:
        return LegacyScheduleSampler(schedule)

    name = str(scfg.get("name", "legacy")).lower()
    if name in {"legacy", "schedule"}:
        return LegacyScheduleSampler(schedule)
    if name in {"uniform", "uniform_continuous", "continuous"}:
        return UniformContinuousSampler(
            min_t=float(scfg.get("min_t", 1e-5)),
            max_t=float(scfg.get("max_t", 1.0)),
        )
    if name in {"uniform_discrete", "discrete"}:
        return UniformDiscreteSampler(
            num_steps=int(scfg.get("num_steps", getattr(schedule, "num_steps", 1000))),
            min_step=int(scfg.get("min_step", 1)),
            max_step=scfg.get("max_step", None),
        )
    if name in {"logit_normal", "logitnormal"}:
        return LogitNormalSampler(
            mean=float(scfg.get("mean", 0.0)),
            std=float(scfg.get("std", 1.0)),
            min_t=float(scfg.get("min_t", 1e-5)),
            max_t=float(scfg.get("max_t", 1.0 - 1e-5)),
        )
    if name == "beta":
        return BetaSampler(
            alpha=float(scfg.get("alpha", 1.0)),
            beta=float(scfg.get("beta", 1.0)),
            min_t=float(scfg.get("min_t", 1e-5)),
            max_t=float(scfg.get("max_t", 1.0)),
        )
    raise ValueError(f"Unknown time_sampler.name={name!r}")
