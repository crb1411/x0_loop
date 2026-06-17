from __future__ import annotations

import re

import torch
import torch.nn.functional as F


VALID_LOSS_TARGETS = {"eps", "x0", "v", "mudata"}
VALID_FORMULAS = {"mse", "l1", "huber", "block_dual_softmax_kl"}


def normalize_loss_target(target: str) -> str:
    target = str(target).lower()
    if target in {"u", "flow", "flow_velocity", "velocity"}:
        target = "v"
    if target not in VALID_LOSS_TARGETS:
        raise ValueError(f"target must be eps | x0 | v | mudata, got {target!r}")
    return target


def _flatten(x: torch.Tensor) -> torch.Tensor:
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


def _sample_minmax_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-example min-max normalization over C,H,W."""
    x_float = x.float()
    x_min = x_float.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x_float.amax(dim=(1, 2, 3), keepdim=True)
    return (x_float - x_min) / (x_max - x_min + float(eps))


def _block_logits(
    x01: torch.Tensor,
    *,
    block_size: int,
    temperature: float,
    channel_reduce: str,
) -> torch.Tensor:
    if x01.ndim != 4:
        raise ValueError(f"block loss expects [B,C,H,W], got shape={tuple(x01.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    h, w = x01.shape[-2:]
    if h < block_size or w < block_size:
        raise ValueError(f"block_size={block_size} is larger than image spatial shape {(h, w)}")

    if h % block_size != 0 or w % block_size != 0:
        h_keep = (h // block_size) * block_size
        w_keep = (w // block_size) * block_size
        x01 = x01[..., :h_keep, :w_keep]

    score = F.avg_pool2d(x01, kernel_size=block_size, stride=block_size)

    channel_reduce = str(channel_reduce).lower()
    if channel_reduce == "mean":
        score = score.mean(dim=1, keepdim=True)
    elif channel_reduce == "sum":
        score = score.sum(dim=1, keepdim=True)
    elif channel_reduce in {"none", "flatten"}:
        pass
    else:
        raise ValueError(f"channel_reduce must be mean | sum | none, got {channel_reduce!r}")

    return score.flatten(1) / float(temperature)


def _kl_from_logits(target_logits: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    target_prob = F.softmax(target_logits.detach(), dim=-1)
    pred_log_prob = F.log_softmax(pred_logits, dim=-1)
    return F.kl_div(pred_log_prob, target_prob, reduction="none").sum(dim=-1)


def block_dual_softmax_kl(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    """Dual bright/dark block-distribution KL, shape [B].

    pred and target are min-max normalized independently per sample.
    Bright branch uses z; dark branch uses 1-z.
    """
    pred01 = _sample_minmax_norm(pred, eps=eps)
    target01 = _sample_minmax_norm(target.detach(), eps=eps)

    pred_bright = _block_logits(pred01, block_size=block_size, temperature=temperature, channel_reduce=channel_reduce)
    target_bright = _block_logits(target01, block_size=block_size, temperature=temperature, channel_reduce=channel_reduce)

    pred_dark = _block_logits(1.0 - pred01, block_size=block_size, temperature=temperature, channel_reduce=channel_reduce)
    target_dark = _block_logits(1.0 - target01, block_size=block_size, temperature=temperature, channel_reduce=channel_reduce)

    return _kl_from_logits(target_bright, pred_bright) + _kl_from_logits(target_dark, pred_dark)


def match_formula(
    formula: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    if formula in {"mse", "l1", "huber"}:
        return regress(formula, pred, target, delta)
    if formula == "block_dual_softmax_kl":
        return block_dual_softmax_kl(
            pred,
            target,
            block_size=block_size,
            temperature=temperature,
            eps=eps,
            channel_reduce=channel_reduce,
        )
    raise ValueError(f"Unknown formula: {formula!r}. Use {' | '.join(sorted(VALID_FORMULAS))}.")


def x0_loss(
    process,
    fb,
    out,
    *,
    formula: str = "mse",
    delta: float = 1.0,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    pred = process.x0_from_output(fb.xt, fb.t, out, aux={})
    return match_formula(
        formula,
        pred,
        process.x0_target(fb),
        delta=delta,
        block_size=block_size,
        temperature=temperature,
        eps=eps,
        channel_reduce=channel_reduce,
    )


def endpoint_loss(
    process,
    fb,
    out,
    *,
    formula: str = "mse",
    delta: float = 1.0,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    pred = process.endpoint_from_output(fb.xt, fb.t, out, aux={})
    return match_formula(
        formula,
        pred,
        process.endpoint_target(fb),
        delta=delta,
        block_size=block_size,
        temperature=temperature,
        eps=eps,
        channel_reduce=channel_reduce,
    )


def v_loss(
    process,
    fb,
    out,
    *,
    formula: str = "mse",
    delta: float = 1.0,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    pred = process.v_from_output(fb.xt, fb.t, out, aux={})
    return match_formula(
        formula,
        pred,
        process.v_target(fb),
        delta=delta,
        block_size=block_size,
        temperature=temperature,
        eps=eps,
        channel_reduce=channel_reduce,
    )


def mudata_loss(
    process,
    fb,
    out,
    *,
    formula: str = "mse",
    delta: float = 1.0,
    block_size: int = 8,
    temperature: float = 0.5,
    eps: float = 1e-6,
    channel_reduce: str = "mean",
) -> torch.Tensor:
    pred = process.mudata_from_output(fb.xt, fb.t, out, aux={})
    return match_formula(
        formula,
        pred,
        process.mudata_target(fb),
        delta=delta,
        block_size=block_size,
        temperature=temperature,
        eps=eps,
        channel_reduce=channel_reduce,
    )


LOSS_FUNCTIONS = {
    "x0": x0_loss,
    "eps": endpoint_loss,
    "v": v_loss,
    "mudata": mudata_loss,
}


def time_weight(weight_fn, fb, like: torch.Tensor) -> torch.Tensor:
    if weight_fn is None:
        return torch.ones_like(like)
    w = weight_fn(fb.t, None)
    if w.ndim > 1:
        w = w.view(w.shape[0], -1).mean(dim=1)
    return w.to(device=like.device, dtype=like.dtype)


def _safe_metric_name(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).lower()).strip("_")


class AtomicLoss:
    def __init__(
        self,
        *,
        target: str,
        formula: str = "mse",
        delta: float = 1.0,
        weight_fn=None,
        coef: float = 1.0,
        block_size: int = 8,
        temperature: float = 0.5,
        eps: float = 1e-6,
        channel_reduce: str = "mean",
        name: str | None = None,
    ):
        self.target = normalize_loss_target(target)
        formula = str(formula).lower()
        if formula not in VALID_FORMULAS:
            raise ValueError(f"formula must be {' | '.join(sorted(VALID_FORMULAS))}, got {formula!r}")
        self.loss_fn = LOSS_FUNCTIONS[self.target]
        self.formula = formula
        self.delta = float(delta)
        self.weight_fn = weight_fn
        self.coef = float(coef)
        self.block_size = int(block_size)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.channel_reduce = str(channel_reduce).lower()
        self.name = _safe_metric_name(name) if name else _safe_metric_name(f"{self.target}_{self.formula}")

    def raw_loss(self, process, fb, out) -> torch.Tensor:
        return self.loss_fn(
            process,
            fb,
            out,
            formula=self.formula,
            delta=self.delta,
            block_size=self.block_size,
            temperature=self.temperature,
            eps=self.eps,
            channel_reduce=self.channel_reduce,
        )

    def weight_term(self, raw: torch.Tensor, fb) -> torch.Tensor:
        return self.coef * time_weight(self.weight_fn, fb, raw) * raw

    def __call__(self, process, fb, out) -> torch.Tensor:
        raw = self.raw_loss(process, fb, out)
        return self.weight_term(raw, fb).mean()

    def __repr__(self) -> str:
        w = "none" if self.weight_fn is None else "weighted"
        opts = ""
        if self.formula == "block_dual_softmax_kl":
            opts = f", block_size={self.block_size}, temperature={self.temperature}, channel_reduce={self.channel_reduce}"
        return f"AtomicLoss(target={self.target}, formula={self.formula}, weight={w}, coef={self.coef}{opts})"


class CompositeLoss:
    def __init__(self, atoms: list[AtomicLoss], *, outer_weight_fn=None):
        if not atoms:
            raise ValueError("CompositeLoss requires at least one atom")
        self.atoms = atoms
        self.outer_weight_fn = outer_weight_fn

    def outer_weight(self, fb, like: torch.Tensor) -> torch.Tensor:
        return time_weight(self.outer_weight_fn, fb, like)

    def __call__(self, process, fb, out) -> dict[str, torch.Tensor]:
        raw_by_target: dict[str, list[torch.Tensor]] = {}
        raw_by_name: dict[str, torch.Tensor] = {}
        contributions: list[torch.Tensor] = []

        for index, atom in enumerate(self.atoms):
            raw = atom.raw_loss(process, fb, out)
            contributions.append(atom.weight_term(raw, fb))
            raw_by_target.setdefault(atom.target, []).append(raw)

            key = f"loss_{atom.name}"
            if key in raw_by_name:
                key = f"{key}_{index}"
            raw_by_name[key] = raw.mean()

        if not contributions:
            raise ValueError("CompositeLoss has no active terms")

        inner = sum(contributions)
        weight = self.outer_weight(fb, inner)
        total = weight * inner

        result = {
            "total": total.mean(),
            "loss_weighted": total.mean(),
            "loss_no_weight": inner.mean(),
            "weight": weight.mean(),
        }
        for target, values in raw_by_target.items():
            result[f"loss_{target}"] = sum(values).mean()
        result.update(raw_by_name)
        return result

    def __repr__(self) -> str:
        w = "none" if self.outer_weight_fn is None else "weighted"
        return f"CompositeLoss(outer_weight={w}, atoms={self.atoms})"
