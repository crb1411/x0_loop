from __future__ import annotations

import torch
import torch.nn.functional as F


VALID_LOSS_TARGETS = {"eps", "x0", "v", "mudata"}


def normalize_loss_target(target: str) -> str:
    target = str(target).lower()
    if target in {"u", "flow", "flow_velocity", "velocity"}:
        target = "v"
    if target not in VALID_LOSS_TARGETS:
        raise ValueError(f"target must be eps | x0 | v | mudata, got {target!r}")
    return target


def _flatten(x: torch.Tensor) -> torch.Tensor:
    # Reduce a [B, ...] tensor to a per-example vector [B].
    return x.view(x.shape[0], -1).mean(dim=1)


def regress(formula: str, pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Unweighted regression error between pred and target, shape [B]."""
    if formula == "mse":
        return _flatten((pred - target).pow(2))
    if formula == "l1":
        return _flatten((pred - target).abs())
    if formula == "huber":
        return _flatten(F.huber_loss(pred, target, delta=delta, reduction="none"))
    raise ValueError(f"Unknown formula: {formula!r}. Use mse | l1 | huber.")


# --------------------------------------------------------------------------- #
# One function per loss type. Each returns the per-example loss [B] in its own
# space; no shared dispatch. Add a new loss kind = add a function + register it.
# --------------------------------------------------------------------------- #

def x0_loss(process, fb, out, *, formula: str = "mse", delta: float = 1.0) -> torch.Tensor:
    """Loss on the clean image x0."""
    pred = process.x0_from_output(fb.xt, fb.t, out, aux={})
    return regress(formula, pred, process.x0_target(fb), delta)


def endpoint_loss(process, fb, out, *, formula: str = "mse", delta: float = 1.0) -> torch.Tensor:
    """Loss on the path endpoint (Gaussian noise, or the learned terminal z)."""
    pred = process.endpoint_from_output(fb.xt, fb.t, out, aux={})
    return regress(formula, pred, process.endpoint_target(fb), delta)


def v_loss(process, fb, out, *, formula: str = "mse", delta: float = 1.0) -> torch.Tensor:
    """Loss on the velocity v = endpoint - x0."""
    pred = process.v_from_output(fb.xt, fb.t, out, aux={})
    return regress(formula, pred, process.v_target(fb), delta)


def mudata_loss(process, fb, out, *, formula: str = "mse", delta: float = 1.0) -> torch.Tensor:
    """Loss on the learnable-endpoint mean head (mudata)."""
    pred = process.mudata_from_output(fb.xt, fb.t, out, aux={})
    return regress(formula, pred, process.mudata_target(fb), delta)


# Config target token -> its dedicated loss function.
LOSS_FUNCTIONS = {
    "x0": x0_loss,
    "eps": endpoint_loss,
    "v": v_loss,
    "mudata": mudata_loss,
}


def time_weight(weight_fn, fb, like: torch.Tensor) -> torch.Tensor:
    """Evaluate a weight_fn(t) into a per-example weight [B]; ones if weight_fn is None."""
    if weight_fn is None:
        return torch.ones_like(like)
    w = weight_fn(fb.t, None)
    if w.ndim > 1:
        w = w.view(w.shape[0], -1).mean(dim=1)
    return w.to(device=like.device, dtype=like.dtype)


class AtomicLoss:
    """Spec for one loss term: which loss function to use, its coef, and an
    optional per-term time weighting.

        term(t) = coef * weight_fn(t) * loss_fn(process, fb, out)      # shape [B]
    """

    def __init__(self, *, target: str, formula: str = "mse", delta: float = 1.0, weight_fn=None, coef: float = 1.0):
        self.target = normalize_loss_target(target)
        if formula not in {"mse", "l1", "huber"}:
            raise ValueError(f"formula must be mse | l1 | huber, got {formula!r}")
        self.loss_fn = LOSS_FUNCTIONS[self.target]
        self.formula = formula
        self.delta = float(delta)
        self.weight_fn = weight_fn
        self.coef = float(coef)

    def weight_term(self, raw: torch.Tensor, fb) -> torch.Tensor:
        """Apply this term's coef and time weighting to its raw loss [B]."""
        return self.coef * time_weight(self.weight_fn, fb, raw) * raw

    def __call__(self, process, fb, out) -> torch.Tensor:
        raw = self.loss_fn(process, fb, out, formula=self.formula, delta=self.delta)
        return self.weight_term(raw, fb).mean()

    def __repr__(self) -> str:
        w = "none" if self.weight_fn is None else "weighted"
        return f"AtomicLoss(target={self.target}, formula={self.formula}, weight={w}, coef={self.coef})"


class CompositeLoss:
    """Total training loss: a flat sum of loss terms, then one optional global
    time re-weighting.

        inner(t)  = Σ_term  coef * weight_fn(t) * loss_fn_term(...)            # [B]
        weight(t) = outer_weight_fn(t)                  # global re-weight; 1.0 if unset
        total(t)  = weight(t) * inner(t)                                       # [B]

    Returns batch-mean scalars: the optimized `total`, plus `loss_no_weight` /
    `loss_weighted` / `weight` and one unweighted per-target loss
    (`loss_x0` / `loss_eps` / `loss_v` / `loss_mudata`) for logging.
    """

    def __init__(self, atoms: list[AtomicLoss], *, outer_weight_fn=None):
        if not atoms:
            raise ValueError("CompositeLoss requires at least one atom")
        self.atoms = atoms
        self.terms: dict[str, AtomicLoss] = {}
        for atom in atoms:
            if atom.target in self.terms:
                raise ValueError(f"Duplicate loss target {atom.target!r}; one term per target.")
            self.terms[atom.target] = atom
        self.outer_weight_fn = outer_weight_fn

    def outer_weight(self, fb, like: torch.Tensor) -> torch.Tensor:
        return time_weight(self.outer_weight_fn, fb, like)

    def __call__(self, process, fb, out) -> dict[str, torch.Tensor]:
        # Each active target calls its own loss function; raw[k] is the
        # unweighted per-example loss [B], contributions[k] its coef/time-weighted part.
        raw: dict[str, torch.Tensor] = {}
        contributions: list[torch.Tensor] = []
        if "x0" in self.terms:
            raw["x0"] = x0_loss(process, fb, out, formula=self.terms["x0"].formula, delta=self.terms["x0"].delta)
            contributions.append(self.terms["x0"].weight_term(raw["x0"], fb))
        if "eps" in self.terms:
            raw["eps"] = endpoint_loss(process, fb, out, formula=self.terms["eps"].formula, delta=self.terms["eps"].delta)
            contributions.append(self.terms["eps"].weight_term(raw["eps"], fb))
        if "v" in self.terms:
            raw["v"] = v_loss(process, fb, out, formula=self.terms["v"].formula, delta=self.terms["v"].delta)
            contributions.append(self.terms["v"].weight_term(raw["v"], fb))
        if "mudata" in self.terms:
            raw["mudata"] = mudata_loss(process, fb, out, formula=self.terms["mudata"].formula, delta=self.terms["mudata"].delta)
            contributions.append(self.terms["mudata"].weight_term(raw["mudata"], fb))
        if not contributions:
            raise ValueError("CompositeLoss has no active terms")

        inner = sum(contributions)                   # Σ coef · weight_fn(t) · raw   [B]
        weight = self.outer_weight(fb, inner)        # outer_weight_fn(t)            [B]
        total = weight * inner                       # [B]

        result = {
            "total": total.mean(),
            "loss_weighted": total.mean(),
            "loss_no_weight": inner.mean(),
            "weight": weight.mean(),
        }
        if "x0" in raw:
            result["loss_x0"] = raw["x0"].mean()
        if "eps" in raw:
            result["loss_eps"] = raw["eps"].mean()
        if "v" in raw:
            result["loss_v"] = raw["v"].mean()
        if "mudata" in raw:
            result["loss_mudata"] = raw["mudata"].mean()
        return result

    def __repr__(self) -> str:
        w = "none" if self.outer_weight_fn is None else "weighted"
        return f"CompositeLoss(outer_weight={w}, atoms={self.atoms})"
