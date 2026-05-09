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
    """One training term: target × formula × t_weight × coef."""

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
        """Returns (unweighted [B], weight [B])."""
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
    """Weighted sum of AtomicLoss terms."""

    def __init__(self, atoms: list[AtomicLoss]):
        if not atoms:
            raise ValueError("CompositeLoss requires at least one atom")
        self.atoms = atoms

    def __call__(self, process, fb, out) -> dict[str, torch.Tensor]:
        """Returns {'total': scalar} plus per-target scalars (e.g. 'eps', 'x0', 'v')."""
        total: torch.Tensor | None = None
        by_target: dict[str, torch.Tensor] = {}
        for atom in self.atoms:
            term = atom(process, fb, out)
            prev = by_target.get(atom.target)
            by_target[atom.target] = (atom.coef * term) if prev is None else (prev + atom.coef * term)
            total = (atom.coef * term) if total is None else (total + atom.coef * term)
        result: dict[str, torch.Tensor] = {"total": total}  # type: ignore[assignment]
        result.update(by_target)
        return result

    def __repr__(self) -> str:
        return f"CompositeLoss({self.atoms})"
