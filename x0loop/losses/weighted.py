from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


WeightFn = Callable[[torch.Tensor, dict | None], torch.Tensor]


@dataclass(frozen=True)
class WeightOptions:
    eps: float = 1e-8
    balance_factor: float = 0.5
    balance_time: str = "auto"
    balance_integral_steps: int = 2000
    target: str | None = None
    floor: float = 0.0
    power: float = 2.0
    gamma: float = 5.0
    normalize: str = "mean"


_ALIASES = {
    "t_x0": "x0",
    "x0_t": "x0",
    "t_eps": "eps",
    "eps_t": "eps",
    "t_mid": "v",
    "v_t": "v",
    "target_default": "target",
    "auto_target": "target",
}
_POLY_MODES = {"x0", "eps", "v"}
_SCHEDULE_MODES = {"snr", "inv_snr", "logsnr", "min_snr"}


def _options(
    *,
    eps: float,
    balance_factor: float,
    balance_time: str,
    balance_integral_steps: int,
    target: str | None,
    floor: float,
    power: float,
    gamma: float,
    normalize: str,
) -> WeightOptions:
    opts = WeightOptions(
        eps=float(eps),
        balance_factor=float(balance_factor),
        balance_time=str(balance_time).lower(),
        balance_integral_steps=int(balance_integral_steps),
        target=None if target is None else str(target).lower(),
        floor=float(floor),
        power=float(power),
        gamma=float(gamma),
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
    if not (0.0 <= opts.balance_factor <= 1.0):
        raise ValueError(f"balance_factor must be in [0, 1], got {opts.balance_factor}")
    if opts.balance_time not in {"auto", "discrete", "continuous"}:
        raise ValueError(f"balance_time must be one of auto/discrete/continuous, got {opts.balance_time}")
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


def _poly_weight(t: torch.Tensor, mode: str, opts: WeightOptions) -> torch.Tensor:
    t = t.float().clamp(0.0, 1.0)
    bases = {
        "x0": 1.0 - t,
        "eps": t,
        "v": 1.0 - (2.0 * t - 1.0).abs(),
    }
    raw = opts.floor + (1.0 - opts.floor) * bases[mode].clamp(0.0, 1.0).pow(opts.power)
    return _normalize_poly(raw, opts)


def _schedule_weight(t: torch.Tensor, mode: str, schedule, opts: WeightOptions) -> torch.Tensor:
    snr = schedule.snr(t).clamp_min(opts.eps)
    ops = {
        "snr": lambda: snr,
        "inv_snr": lambda: 1.0 / snr,
        "logsnr": lambda: torch.log(snr),
        "min_snr": lambda: torch.minimum(snr, torch.full_like(snr, opts.gamma)) / snr,
    }
    return ops[mode]()


def _target_weight(t: torch.Tensor, opts: WeightOptions) -> torch.Tensor:
    if opts.target in _POLY_MODES:
        return _poly_weight(t, opts.target, opts)
    return torch.ones_like(t, dtype=torch.float32)


def _make_balance_weight(schedule, opts: WeightOptions) -> WeightFn:
    caches: dict[str, dict[tuple[torch.device, torch.dtype], torch.Tensor]] = {
        "discrete": {},
        "continuous": {},
    }

    def average_inv_alpha(kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = caches[kind]
        key = (device, dtype)
        if key not in cache:
            if kind == "discrete":
                steps = int(getattr(schedule, "num_steps", 1000))
                grid = torch.arange(1, steps + 1, device=device, dtype=torch.float32) / float(steps)
            else:
                steps = opts.balance_integral_steps
                grid = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / float(steps)
            inv_alpha = 1.0 / schedule.alpha(grid).clamp_min(opts.eps)
            cache[key] = inv_alpha.to(dtype).mean().clamp_min(opts.eps)
        return cache[key]

    def choose_kind(t: torch.Tensor) -> str:
        if opts.balance_time != "auto":
            return opts.balance_time
        steps = float(int(getattr(schedule, "num_steps", 1000)))
        is_discrete = torch.allclose(t * steps, torch.round(t * steps), atol=1e-6, rtol=0.0)
        return "discrete" if is_discrete else "continuous"

    def weight(t: torch.Tensor, aux=None) -> torch.Tensor:
        del aux
        alpha_t = schedule.alpha(t).clamp_min(opts.eps)
        inv_alpha_avg = average_inv_alpha(choose_kind(t), device=t.device, dtype=alpha_t.dtype)
        return (1.0 - opts.balance_factor) + opts.balance_factor * (alpha_t * inv_alpha_avg)

    return weight


def make_weight_fn(
    name: str,
    schedule,
    eps: float = 1e-8,
    balance_factor: float = 0.5,
    balance_time: str = "auto",
    balance_integral_steps: int = 2000,
    target: str | None = None,
    floor: float = 0.0,
    power: float = 2.0,
    gamma: float = 5.0,
    normalize: str = "mean",
) -> WeightFn:
    opts = _options(
        eps=eps,
        balance_factor=balance_factor,
        balance_time=balance_time,
        balance_integral_steps=balance_integral_steps,
        target=target,
        floor=floor,
        power=power,
        gamma=gamma,
        normalize=normalize,
    )
    mode = _ALIASES.get(str(name).lower(), str(name).lower())

    factories: dict[str, Callable[[], WeightFn]] = {
        **{m: (lambda m=m: _drop_aux(lambda t, m=m: _poly_weight(t, m, opts))) for m in _POLY_MODES},
        **{m: (lambda m=m: _drop_aux(lambda t, m=m: _schedule_weight(t, m, schedule, opts))) for m in _SCHEDULE_MODES},
        "target": lambda: _drop_aux(lambda t: _target_weight(t, opts)),
        "balance_weights": lambda: _make_balance_weight(schedule, opts),
    }
    try:
        return factories[mode]()
    except KeyError as exc:
        raise ValueError(f"Unknown weight fn: {name}") from exc


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
