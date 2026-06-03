from __future__ import annotations

import torch
import torch.nn.functional as F


VALID_LOSS_TARGETS = {"eps", "x0", "v"}


def normalize_loss_target(target: str) -> str:
    target = str(target).lower()
    if target in {"u", "flow", "flow_velocity", "velocity"}:
        target = "v"
    if target not in VALID_LOSS_TARGETS:
        raise ValueError(f"target must be eps | x0 | v, got {target!r}")
    return target


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
        self.target = normalize_loss_target(target)
        if formula not in {"mse", "l1", "huber"}:
            raise ValueError(f"formula must be mse | l1 | huber, got {formula!r}")
        self.formula = formula
        self.delta = float(delta)
        self.weight_fn = weight_fn
        self.coef = float(coef)

    def _pred_and_target(self, process, fb, out):
        if self.target == "eps":
            return process.eps_from_output(fb.xt, fb.t, out, aux=fb.aux), process.eps_target(fb)
        if self.target == "x0":
            return process.x0_from_output(fb.xt, fb.t, out, aux=fb.aux), process.x0_target(fb)
        if self.target == "v":
            return process.v_from_output(fb.xt, fb.t, out, aux=fb.aux), process.v_target(fb)
        raise AssertionError(f"Unexpected loss target={self.target!r}")

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
    """Weighted sum of AtomicLoss terms."""

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
        inner: torch.Tensor | None = None
        by_target_raw: dict[str, torch.Tensor] = {}
        by_target_no_weight: dict[str, torch.Tensor] = {}
        for atom in self.atoms:
            raw, per_space_w = atom.per_example(process, fb, out)
            term_no_outer_weight = atom.coef * per_space_w * raw
            by_target_raw[atom.target] = raw if atom.target not in by_target_raw else by_target_raw[atom.target] + raw
            by_target_no_weight[atom.target] = (
                term_no_outer_weight
                if atom.target not in by_target_no_weight
                else by_target_no_weight[atom.target] + term_no_outer_weight
            )
            inner = term_no_outer_weight if inner is None else inner + term_no_outer_weight
        assert inner is not None
        outer_w = self.outer_weight(fb, inner)
        total = outer_w * inner
        result: dict[str, torch.Tensor] = {
            "loss_no_weight": inner,
            "loss_weighted": total,
            "inner": inner,
            "weight": outer_w,
            "total": total,
        }
        for k, v in by_target_no_weight.items():
            result[k] = outer_w * v
            result[f"{k}_no_weight"] = v
            result[f"{k}_weighted"] = outer_w * v
        for k, v in by_target_raw.items():
            result[f"{k}_raw"] = v
        return result

    def __call__(self, process, fb, out) -> dict[str, torch.Tensor]:
        per_ex = self.per_example(process, fb, out)
        return {k: v.mean() for k, v in per_ex.items()}

    def __repr__(self) -> str:
        w = "none" if self.outer_weight_fn is None else "weighted"
        return f"CompositeLoss(outer_weight={w}, atoms={self.atoms})"
