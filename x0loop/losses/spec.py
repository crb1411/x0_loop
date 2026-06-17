from __future__ import annotations

from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.losses.weighted import make_weight_fn


def _make_weight_fn(cfg: dict, schedule, *, key: str):
    weight = str(cfg.get(key, "none")).lower()
    if weight == "none":
        return None
    prefix = f"{key}_"
    return make_weight_fn(
        weight,
        schedule=schedule,
        balance_integral_steps=int(cfg.get("balance_integral_steps", 2000)),
        floor=float(cfg.get("weight_floor", cfg.get("outer_weight_floor", 0.0))),
        power=float(cfg.get("weight_power", cfg.get("outer_weight_power", 0.5))),
        gamma=float(cfg.get("min_snr_gamma", cfg.get("gamma", 5.0))),
        skew=float(cfg.get(f"{prefix}skew", cfg.get("weight_skew", cfg.get("skew", 0.0)))),
        p2_k=float(cfg.get(f"{prefix}p2_k", cfg.get("p2_k", 1.0))),
        p2_gamma=float(cfg.get(f"{prefix}p2_gamma", cfg.get("p2_gamma", 1.0))),
        sigma_data=float(cfg.get(f"{prefix}sigma_data", cfg.get("sigma_data", 0.5))),
    )


def _build_atom(term: dict, schedule) -> AtomicLoss:
    target = str(term.get("target", "eps")).lower()
    formula = str(term.get("formula", "mse")).lower()
    delta = float(term.get("delta", 1.0))
    coef = float(term.get("coef", 1.0))

    # Per-space weighting is kept only as an explicit escape hatch.
    weight_fn = _make_weight_fn(term, schedule, key="weight")
    return AtomicLoss(
        target=target,
        formula=formula,
        delta=delta,
        weight_fn=weight_fn,
        coef=coef,
        block_size=int(term.get("block_size", 8)),
        temperature=float(term.get("temperature", 0.5)),
        eps=float(term.get("eps", 1e-6)),
        channel_reduce=str(term.get("channel_reduce", "mean")).lower(),
        name=term.get("name", None),
    )


def build_loss(cfg_loss: dict, schedule) -> CompositeLoss:
    """Build CompositeLoss from the `loss:` config section.

    Preferred outer-weight schema:
        loss:
          outer_weight: triangular  # none | triangular | skew_triangular | p2 | min_snr | edm
          outer_weight_power: 0.5
          outer_weight_floor: 0.0
          terms:
            - {target: x0, formula: mse, coef: 1.0}
            - {target: x0, formula: block_dual_softmax_kl, coef: 0.01, block_size: 8, temperature: 0.5}

    Backward-compatible per-term schema:
        loss:
          terms:
            - {target: eps, formula: mse, coef: 1.0, weight: min_snr}
    """
    if "terms" in cfg_loss:
        atoms = [_build_atom(t, schedule) for t in cfg_loss["terms"]]
    else:
        atoms = [_build_atom(cfg_loss, schedule)]

    outer_weight_fn = _make_weight_fn(cfg_loss, schedule, key="outer_weight")
    return CompositeLoss(atoms, outer_weight_fn=outer_weight_fn)
