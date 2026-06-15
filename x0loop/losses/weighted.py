from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


WeightFn = Callable[[torch.Tensor, dict | None], torch.Tensor]


@dataclass(frozen=True)
class WeightOptions:
    eps: float = 1e-8
    balance_integral_steps: int = 2000
    floor: float = 0.0
    power: float = 2.0
    gamma: float = 5.0
    skew: float = 0.0
    p2_k: float = 1.0
    p2_gamma: float = 1.0
    sigma_data: float = 0.5
    normalize: str = "mean"


_WEIGHT_MODES = ("none", "triangular", "skew_triangular", "p2", "min_snr", "edm")


def _options(
    *,
    eps: float,
    balance_integral_steps: int,
    floor: float,
    power: float,
    gamma: float,
    skew: float,
    p2_k: float,
    p2_gamma: float,
    sigma_data: float,
    normalize: str,
) -> WeightOptions:
    opts = WeightOptions(
        eps=float(eps),
        balance_integral_steps=int(balance_integral_steps),
        floor=float(floor),
        power=float(power),
        gamma=float(gamma),
        skew=float(skew),
        p2_k=float(p2_k),
        p2_gamma=float(p2_gamma),
        sigma_data=float(sigma_data),
        normalize=str(normalize).lower(),
    )
    if opts.normalize not in {"none", "mean"}:
        raise ValueError(f"normalize must be none | mean, got {opts.normalize}")
    if opts.floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {opts.floor}")
    if opts.power <= 0.0:
        raise ValueError(f"power must be > 0, got {opts.power}")
    if opts.gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {opts.gamma}")
    if not (-1.0 <= opts.skew <= 1.0):
        raise ValueError(f"skew must be in [-1, 1] to keep skew_triangular non-negative, got {opts.skew}")
    if opts.p2_k <= 0.0:
        raise ValueError(f"p2_k must be > 0, got {opts.p2_k}")
    if opts.p2_gamma <= 0.0:
        raise ValueError(f"p2_gamma must be > 0, got {opts.p2_gamma}")
    if opts.sigma_data <= 0.0:
        raise ValueError(f"sigma_data must be > 0, got {opts.sigma_data}")
    if opts.balance_integral_steps <= 0:
        raise ValueError("balance_integral_steps must be > 0")
    return opts


def _drop_aux(fn: Callable[[torch.Tensor], torch.Tensor]) -> WeightFn:
    def wrapped(t: torch.Tensor, aux=None) -> torch.Tensor:
        del aux
        return fn(t)

    return wrapped


def _normalize_poly(w: torch.Tensor, opts: WeightOptions) -> torch.Tensor:
    if opts.normalize == "none":
        return w
    mean = opts.floor + (1.0 - opts.floor) / (opts.power + 1.0)
    return w / max(mean, opts.eps)


def _normalize_by_uniform_grid(
    fn: Callable[[torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    opts: WeightOptions,
) -> torch.Tensor:
    weight = fn(t)
    if opts.normalize == "none":
        return weight
    steps = int(opts.balance_integral_steps)
    grid = (torch.arange(steps, device=t.device, dtype=torch.float32) + 0.5) / float(steps)
    mean = fn(grid).to(torch.float64).mean().clamp_min(opts.eps)
    return weight / mean.to(device=t.device, dtype=weight.dtype)


def _triangular_raw(t: torch.Tensor, opts: WeightOptions) -> torch.Tensor:
    t = t.float().clamp(0.0, 1.0)
    base = 1.0 - (2.0 * t - 1.0).abs()
    return opts.floor + (1.0 - opts.floor) * base.clamp(0.0, 1.0).pow(opts.power)


def _triangular_weight(t: torch.Tensor, opts: WeightOptions) -> torch.Tensor:
    raw = _triangular_raw(t, opts)
    return _normalize_poly(raw, opts)


def _skew_triangular_weight(t: torch.Tensor, opts: WeightOptions) -> torch.Tensor:
    def raw(x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0.0, 1.0)
        skew_factor = 1.0 + opts.skew * (2.0 * x - 1.0)
        return _triangular_raw(x, opts) * skew_factor

    return _normalize_by_uniform_grid(raw, t, opts)


def _p2_weight(t: torch.Tensor, schedule, opts: WeightOptions) -> torch.Tensor:
    def raw(x: torch.Tensor) -> torch.Tensor:
        snr = schedule.snr(x).clamp_min(opts.eps)
        return (opts.p2_k + snr).pow(-opts.p2_gamma)

    return _normalize_by_uniform_grid(raw, t, opts)


def _min_snr_weight(t: torch.Tensor, schedule, opts: WeightOptions) -> torch.Tensor:
    def raw(x: torch.Tensor) -> torch.Tensor:
        snr = schedule.snr(x).clamp_min(opts.eps)
        return torch.minimum(snr, torch.full_like(snr, opts.gamma)) / snr

    return _normalize_by_uniform_grid(raw, t, opts)


def _edm_weight(t: torch.Tensor, schedule, opts: WeightOptions) -> torch.Tensor:
    def raw(x: torch.Tensor) -> torch.Tensor:
        sigma = (schedule.sigma(x) / schedule.alpha(x).clamp_min(opts.eps)).clamp_min(opts.eps)
        sigma_data = torch.full_like(sigma, opts.sigma_data)
        return (sigma.square() + sigma_data.square()) / (sigma * sigma_data).square().clamp_min(opts.eps)

    return _normalize_by_uniform_grid(raw, t, opts)


def make_weight_fn(
    name: str,
    schedule,
    eps: float = 1e-8,
    balance_integral_steps: int = 2000,
    floor: float = 0.0,
    power: float = 2.0,
    gamma: float = 5.0,
    skew: float = 0.0,
    p2_k: float = 1.0,
    p2_gamma: float = 1.0,
    sigma_data: float = 0.5,
    normalize: str = "mean",
) -> WeightFn:
    opts = _options(
        eps=eps,
        balance_integral_steps=balance_integral_steps,
        floor=floor,
        power=power,
        gamma=gamma,
        skew=skew,
        p2_k=p2_k,
        p2_gamma=p2_gamma,
        sigma_data=sigma_data,
        normalize=normalize,
    )
    mode = str(name).lower()

    factories: dict[str, Callable[[], WeightFn]] = {
        "none": lambda: _drop_aux(lambda t: torch.ones_like(t, dtype=torch.float32)),
        "triangular": lambda: _drop_aux(lambda t: _triangular_weight(t, opts)),
        "skew_triangular": lambda: _drop_aux(lambda t: _skew_triangular_weight(t, opts)),
        "p2": lambda: _drop_aux(lambda t: _p2_weight(t, schedule, opts)),
        "min_snr": lambda: _drop_aux(lambda t: _min_snr_weight(t, schedule, opts)),
        "edm": lambda: _drop_aux(lambda t: _edm_weight(t, schedule, opts)),
    }
    try:
        return factories[mode]()
    except KeyError as exc:
        allowed = " | ".join(_WEIGHT_MODES)
        raise ValueError(f"Unknown weight fn: {name}. Expected one of: {allowed}") from exc


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
