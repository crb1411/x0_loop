from __future__ import annotations

from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.losses.weighted import make_weight_fn


def _build_atom(term: dict, schedule) -> AtomicLoss:
    target = str(term.get("target", "eps")).lower()
    formula = str(term.get("formula", "mse")).lower()
    delta = float(term.get("delta", 1.0))
    weight = str(term.get("weight", "none")).lower()
    coef = float(term.get("coef", 1.0))

    weight_fn = None
    if weight != "none":
        weight_fn = make_weight_fn(
            weight,
            schedule=schedule,
            balance_factor=float(term.get("balance_factor", 0.5)),
            balance_time=str(term.get("balance_time", "auto")),
            balance_integral_steps=int(term.get("balance_integral_steps", 2000)),
        )

    return AtomicLoss(target=target, formula=formula, delta=delta, weight_fn=weight_fn, coef=coef)


def build_loss(cfg_loss: dict, schedule) -> CompositeLoss:
    """Build CompositeLoss from the `loss:` config section.

    Single term (no `terms` key):
        loss: {target: eps, formula: mse, weight: none}

    Multiple terms:
        loss:
          terms:
            - {target: eps, formula: mse, coef: 0.5}
            - {target: x0,  formula: mse, coef: 0.5}
    """
    if "terms" in cfg_loss:
        atoms = [_build_atom(t, schedule) for t in cfg_loss["terms"]]
    else:
        atoms = [_build_atom(cfg_loss, schedule)]
    return CompositeLoss(atoms)
