from __future__ import annotations

import ast

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


class FlowSolverGridSampler(TimeSampler):
    def __init__(
        self,
        *,
        steps: int | list[int] | tuple[int, ...],
        min_t: float = 1e-5,
        include_t0: bool = False,
    ):
        steps = self._coerce_steps(steps)
        self.steps = [int(step) for step in steps]
        self.min_t = float(min_t)
        self.include_t0 = bool(include_t0)
        if not self.steps:
            raise ValueError("steps must contain at least one positive integer")
        if any(step <= 0 for step in self.steps):
            raise ValueError(f"steps must be positive, got {self.steps}")

        values: list[float] = []
        for step_count in self.steps:
            start = 0 if self.include_t0 else 1
            values.extend(float(i) / float(step_count) for i in range(start, step_count + 1))
        self.values = [max(self.min_t, min(1.0, value)) for value in values]

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(0, len(self.values), (batch_size,), device=device)
        values = torch.tensor(self.values, device=device, dtype=torch.float32)
        return values[idx]

    @staticmethod
    def _coerce_steps(steps) -> list[int] | list[str]:
        if isinstance(steps, int):
            return [steps]
        if isinstance(steps, str):
            text = steps.strip()
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
            if isinstance(parsed, int):
                return [parsed]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        return list(steps)


class MixedTimeSampler(TimeSampler):
    def __init__(self, *, base: TimeSampler, branch: TimeSampler, branch_prob: float):
        self.base = base
        self.branch = branch
        self.branch_prob = float(branch_prob)
        if not (0.0 <= self.branch_prob <= 1.0):
            raise ValueError(f"branch_prob must be in [0, 1], got {branch_prob}")

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base_t = self.base.sample(batch_size, device=device)
        if self.branch_prob == 0.0:
            return base_t
        branch_t = self.branch.sample(batch_size, device=device)
        if self.branch_prob == 1.0:
            return branch_t
        use_branch = torch.rand(batch_size, device=device) < self.branch_prob
        return torch.where(use_branch, branch_t, base_t)


def _build_base_time_sampler(scfg: dict, schedule) -> TimeSampler:
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


def _grid_mix_prob(scfg: dict) -> float:
    for key in ("grid_mix_prob", "mix_grid_prob", "grid_prob", "p_grid"):
        if key in scfg:
            return float(scfg[key])
    return 0.0


def build_time_sampler(cfg: dict, schedule) -> TimeSampler:
    scfg = cfg.get("time_sampler", None)
    if scfg is None:
        return LegacyScheduleSampler(schedule)

    base = _build_base_time_sampler(scfg, schedule)
    branch_prob = _grid_mix_prob(scfg)
    if not (0.0 <= branch_prob <= 1.0):
        raise ValueError(f"grid_mix_prob must be in [0, 1], got {branch_prob}")
    if branch_prob <= 0.0:
        return base
    if getattr(schedule, "mode", None) != "flow":
        raise ValueError("time_sampler grid mixing is only supported for flow schedules.")
    branch = FlowSolverGridSampler(
        steps=scfg.get("grid_steps", [50, 20]),
        min_t=float(scfg.get("grid_min_t", 1e-5)),
        include_t0=bool(scfg.get("grid_include_t0", False)),
    )
    return MixedTimeSampler(base=base, branch=branch, branch_prob=branch_prob)
