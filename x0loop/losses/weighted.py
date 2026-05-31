from __future__ import annotations

from typing import Callable

import torch


WeightFn = Callable[[torch.Tensor, dict | None], torch.Tensor]


def make_weight_fn(
    name: str,
    schedule,
    eps: float = 1e-8,
    balance_factor: float = 0.5,
    balance_time: str = "auto",
    balance_integral_steps: int = 2000,
    target: str | None = None,
    floor: float = 0.0,
    power: float = 0.5,
    gamma: float = 5.0,
) -> WeightFn:
    name = str(name).lower()
    target = None if target is None else str(target).lower()
    floor = float(floor)
    power = float(power)
    gamma = float(gamma)
    if floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {floor}")
    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    balance_factor = float(balance_factor)
    if not (0.0 <= balance_factor <= 1.0):
        raise ValueError(f"balance_factor must be in [0, 1], got {balance_factor}")
    balance_time = str(balance_time).lower()
    if balance_time not in {"auto", "discrete", "continuous"}:
        raise ValueError(f"balance_time must be one of auto/discrete/continuous, got {balance_time}")
    balance_integral_steps = int(balance_integral_steps)
    if balance_integral_steps <= 0:
        raise ValueError("balance_integral_steps must be > 0")

    def _snr(t, aux=None):
        return schedule.snr(t).clamp_min(eps)

    def _inv_snr(t, aux=None):
        return 1.0 / schedule.snr(t).clamp_min(eps)

    def _logsnr(t, aux=None):
        return torch.log(schedule.snr(t).clamp_min(eps))

    def _min_snr(t, aux=None):
        snr = schedule.snr(t).clamp_min(eps)
        return torch.minimum(snr, torch.full_like(snr, gamma)) / snr

    def _t_x0(t, aux=None):
        del aux
        # x0 is easy near t=0 and hard/noisy near t=1.
        # Default: w(t)=sqrt(1-t). `floor` can keep a non-zero tail if needed.
        return floor + (1.0 - floor) * (1.0 - t.float()).clamp(0.0, 1.0).pow(power)

    def _t_eps(t, aux=None):
        del aux
        return floor + (1.0 - floor) * t.float().clamp(0.0, 1.0).pow(power)

    def _t_mid(t, aux=None):
        del aux
        # Peak at t=0.5, floor at both endpoints.
        mid = 1.0 - (2.0 * t.float().clamp(0.0, 1.0) - 1.0).abs()
        return floor + (1.0 - floor) * mid.clamp(0.0, 1.0).pow(power)

    def _target_default(t, aux=None):
        if target == "x0":
            return _t_x0(t, aux)
        if target == "eps":
            return _t_eps(t, aux)
        if target == "v":
            return _t_mid(t, aux)
        return torch.ones_like(t, dtype=torch.float32)

    inv_alpha_avg_discrete_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
    inv_alpha_avg_continuous_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _inv_alpha_avg_discrete(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key in inv_alpha_avg_discrete_cache:
            return inv_alpha_avg_discrete_cache[key]
        steps = int(getattr(schedule, "num_steps", 1000))
        t_grid = torch.arange(1, steps + 1, device=device, dtype=torch.float32) / float(steps)
        inv_alpha = 1.0 / schedule.alpha(t_grid).clamp_min(eps)
        avg = inv_alpha.to(dtype).mean().clamp_min(eps)
        inv_alpha_avg_discrete_cache[key] = avg
        return avg

    def _inv_alpha_avg_continuous(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key in inv_alpha_avg_continuous_cache:
            return inv_alpha_avg_continuous_cache[key]
        m = balance_integral_steps
        t_mid = (torch.arange(m, device=device, dtype=torch.float32) + 0.5) / float(m)
        inv_alpha = 1.0 / schedule.alpha(t_mid).clamp_min(eps)
        avg = inv_alpha.to(dtype).mean().clamp_min(eps)
        inv_alpha_avg_continuous_cache[key] = avg
        return avg

    def _balance_weights(t, aux=None):
        del aux
        alpha_t = schedule.alpha(t).clamp_min(eps)

        if balance_time == "discrete":
            inv_alpha_avg = _inv_alpha_avg_discrete(device=t.device, dtype=alpha_t.dtype)
        elif balance_time == "continuous":
            inv_alpha_avg = _inv_alpha_avg_continuous(device=t.device, dtype=alpha_t.dtype)
        else:
            steps = float(int(getattr(schedule, "num_steps", 1000)))
            is_discrete_batch = torch.allclose(t * steps, torch.round(t * steps), atol=1e-6, rtol=0.0)
            inv_alpha_avg = (
                _inv_alpha_avg_discrete(device=t.device, dtype=alpha_t.dtype)
                if is_discrete_batch
                else _inv_alpha_avg_continuous(device=t.device, dtype=alpha_t.dtype)
            )

        return (1.0 - balance_factor) + balance_factor * (alpha_t * inv_alpha_avg)

    if name == "snr":
        return _snr
    if name == "inv_snr":
        return _inv_snr
    if name == "logsnr":
        return _logsnr
    if name == "min_snr":
        return _min_snr
    if name in {"t_x0", "x0", "x0_t"}:
        return _t_x0
    if name in {"t_eps", "eps", "eps_t"}:
        return _t_eps
    if name in {"t_mid", "v", "v_t"}:
        return _t_mid
    if name in {"target", "target_default", "auto_target"}:
        return _target_default
    if name == "balance_weights":
        return _balance_weights
    raise ValueError(f"Unknown weight fn: {name}")


class PiecewiseWeightFn:
    def __init__(self, edges: list[float], values: list[float]):
        if len(edges) < 2 or len(values) != len(edges) - 1:
            raise ValueError("values length must be len(edges)-1")
        self.edges = edges
        self.values = values

    def __call__(self, t: torch.Tensor, aux=None) -> torch.Tensor:
        w = torch.zeros_like(t)
        for i, v in enumerate(self.values):
            lo, hi = self.edges[i], self.edges[i + 1]
            if i == len(self.values) - 1:
                mask = (t >= lo) & (t <= hi)
            else:
                mask = (t >= lo) & (t < hi)
            w = torch.where(mask, torch.full_like(t, float(v)), w)
        return w
