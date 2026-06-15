from __future__ import annotations

from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.losses.weighted import make_weight_fn


def _make_weight_fn(cfg: dict, schedule, *, key: str, target: str | None = None):
    weight = str(cfg.get(key, "none")).lower()
    if weight == "none":
        return None
    return make_weight_fn(
        weight,
        schedule=schedule,
        balance_factor=float(cfg.get("balance_factor", 0.5)),
        balance_time=str(cfg.get("balance_time", "auto")),
        balance_integral_steps=int(cfg.get("balance_integral_steps", 2000)),
        target=target,
        floor=float(cfg.get("weight_floor", cfg.get("outer_weight_floor", 0.0))),
        power=float(cfg.get("weight_power", cfg.get("outer_weight_power", 0.5))),
        gamma=float(cfg.get("min_snr_gamma", cfg.get("gamma", 5.0))),
    )


def _build_atom(term: dict, schedule) -> AtomicLoss:
    target = str(term.get("target", "eps")).lower()
    formula = str(term.get("formula", "mse")).lower()
    delta = float(term.get("delta", 1.0))
    coef = float(term.get("coef", 1.0))

    # Per-space weighting is kept only as an explicit escape hatch.
    weight_fn = _make_weight_fn(term, schedule, key="weight", target=target)
    return AtomicLoss(target=target, formula=formula, delta=delta, weight_fn=weight_fn, coef=coef)


def build_loss(cfg_loss: dict, schedule) -> CompositeLoss:
    """Build CompositeLoss from the `loss:` config section.

    Preferred outer-weight schema:
        loss:
          outer_weight: x0      # none | x0 | eps | v | target | snr | inv_snr | logsnr | min_snr
          outer_weight_power: 0.5
          outer_weight_floor: 0.0
          terms:
            - {target: x0, formula: mse, coef: 1.0}

    Backward-compatible per-term schema:
        loss:
          terms:
            - {target: eps, formula: mse, coef: 1.0, weight: snr}
    """
    if "terms" in cfg_loss:
        atoms = [_build_atom(t, schedule) for t in cfg_loss["terms"]]
    else:
        atoms = [_build_atom(cfg_loss, schedule)]

    primary_target = str(cfg_loss["outer_weight_target"]).lower() if "outer_weight_target" in cfg_loss else (atoms[0].target if len(atoms) == 1 else None)
    outer_weight_fn = _make_weight_fn(cfg_loss, schedule, key="outer_weight", target=primary_target)
    return CompositeLoss(atoms, outer_weight_fn=outer_weight_fn)
