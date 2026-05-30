from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).mean(dim=1)


def regress(formula: str, pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Per-example unweighted loss, shape [B]."""
    if formula == "mse":
        return _flatten((pred - target).pow(2))
    if formula == "l1":
        return _flatten((pred - target).abs())
    if formula == "huber":
        return _flatten(F.huber_loss(pred, target, delta=delta, reduction="none"))
    raise ValueError(f"Unknown formula: {formula!r}. Use mse | l1 | huber.")


class AtomicLoss:
    """One training term: target × formula × optional per-space t_weight × coef."""

    def __init__(self, *, target: str, formula: str, delta: float = 1.0, weight_fn=None, coef: float = 1.0):
        if target not in {"eps", "x0", "v"}:
            raise ValueError(f"target must be eps | x0 | v, got {target!r}")
        if formula not in {"mse", "l1", "huber"}:
            raise ValueError(f"formula must be mse | l1 | huber, got {formula!r}")
        self.target = target
        self.formula = formula
        self.delta = float(delta)
        self.weight_fn = weight_fn  # callable(t, aux) -> Tensor[B], or None for uniform
        self.coef = float(coef)

    def _pred_and_target(self, process, fb, out):
        if self.target == "eps":
            return process.eps_from_output(fb.xt, fb.t, out, aux=fb.aux), process.eps_target(fb)
        if self.target == "x0":
            return process.x0_from_output(fb.xt, fb.t, out, aux=fb.aux), process.x0_target(fb)
        return process.v_from_output(fb.xt, fb.t, out, aux=fb.aux), process.v_target(fb)

    def per_example(self, process, fb, out) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (unweighted [B], per-space weight [B])."""
        pred, tgt = self._pred_and_target(process, fb, out)
        unweighted = regress(self.formula, pred, tgt, self.delta)
        if self.weight_fn is not None:
            w = self.weight_fn(fb.t, fb.aux)
            if w.ndim > 1:
                w = w.view(w.shape[0], -1).mean(dim=1)
            w = w.to(dtype=unweighted.dtype)
        else:
            w = torch.ones_like(unweighted)
        return unweighted, w

    def __call__(self, process, fb, out) -> torch.Tensor:
        unweighted, w = self.per_example(process, fb, out)
        return (w * unweighted).mean()

    def __repr__(self) -> str:
        w = "none" if self.weight_fn is None else "weighted"
        return f"AtomicLoss(target={self.target}, formula={self.formula}, weight={w}, coef={self.coef})"


class CompositeLoss:
    """Weighted sum of AtomicLoss terms.

    Default semantics are outer timestep weighting:

        loss_t = outer_weight(t) * sum_i coef_i * loss_i(t)

    AtomicLoss.weight_fn remains available only for explicit per-space weighting.
    """

    def __init__(self, atoms: list[AtomicLoss], *, outer_weight_fn=None):
        if not atoms:
            raise ValueError("CompositeLoss requires at least one atom")
        self.atoms = atoms
        self.outer_weight_fn = outer_weight_fn

    def outer_weight(self, fb, ref: torch.Tensor) -> torch.Tensor:
        if self.outer_weight_fn is None:
            return torch.ones_like(ref)
        w = self.outer_weight_fn(fb.t, fb.aux)
        if w.ndim > 1:
            w = w.view(w.shape[0], -1).mean(dim=1)
        return w.to(device=ref.device, dtype=ref.dtype)

    def per_example(self, process, fb, out) -> dict[str, torch.Tensor]:
        """Returns per-example losses after coef and outer weighting."""
        inner: torch.Tensor | None = None
        by_target_raw: dict[str, torch.Tensor] = {}
        by_target: dict[str, torch.Tensor] = {}

        for atom in self.atoms:
            raw, per_space_w = atom.per_example(process, fb, out)
            term = atom.coef * per_space_w * raw
            by_target_raw[atom.target] = raw if atom.target not in by_target_raw else by_target_raw[atom.target] + raw
            by_target[atom.target] = term if atom.target not in by_target else by_target[atom.target] + term
            inner = term if inner is None else inner + term

        assert inner is not None
        outer_w = self.outer_weight(fb, inner)
        result: dict[str, torch.Tensor] = {
            "inner": inner,
            "weight": outer_w,
            "total": outer_w * inner,
        }
        for k, v in by_target.items():
            result[k] = outer_w * v
        for k, v in by_target_raw.items():
            result[f"{k}_raw"] = v
        return result

    def __call__(self, process, fb, out) -> dict[str, torch.Tensor]:
        """Returns {'total': scalar} plus per-target scalars."""
        per_ex = self.per_example(process, fb, out)
        return {k: v.mean() for k, v in per_ex.items()}

    def __repr__(self) -> str:
        w = "none" if self.outer_weight_fn is None else "weighted"
        return f"CompositeLoss(outer_weight={w}, atoms={self.atoms})"
