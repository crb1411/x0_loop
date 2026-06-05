from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AdversarialConfig:
    enabled: bool = False
    weight: float = 0.0
    start_step: int = 0
    warmup_steps: int = 0
    update_every: int = 1
    d_steps: int = 1
    loss: str = "hinge"
    clamp_fake_for_d: bool = False
    fake_space: str = "x0_hat"
    r1_gamma: float = 0.0
    r1_interval: int = 16


def build_adversarial_config(cfg: dict) -> AdversarialConfig:
    ac = cfg.get("adversarial", {}) or {}
    return AdversarialConfig(
        enabled=bool(ac.get("enabled", False)),
        weight=float(ac.get("weight", 0.0)),
        start_step=int(ac.get("start_step", 0)),
        warmup_steps=int(ac.get("warmup_steps", 0)),
        update_every=int(ac.get("update_every", 1)),
        d_steps=int(ac.get("d_steps", 1)),
        loss=str(ac.get("loss", "hinge")).lower(),
        clamp_fake_for_d=bool(ac.get("clamp_fake_for_d", False)),
        fake_space=str(ac.get("fake_space", "x0_hat")).lower(),
        r1_gamma=float(((ac.get("r1", {}) or {}).get("gamma", ac.get("r1_gamma", 0.0)))),
        r1_interval=int(((ac.get("r1", {}) or {}).get("interval", ac.get("r1_interval", 16)))),
    )


def adversarial_weight(config: AdversarialConfig, step: int) -> float:
    if (not config.enabled) or config.weight <= 0.0 or step < config.start_step:
        return 0.0
    if config.warmup_steps <= 0:
        return float(config.weight)
    progress = min(max((step - config.start_step + 1) / float(config.warmup_steps), 0.0), 1.0)
    return float(config.weight) * progress


def t_weight(t: torch.Tensor, cfg: dict) -> torch.Tensor:
    tw = ((cfg.get("adversarial", {}) or {}).get("t_weight", {}) or {})
    name = str(tw.get("name", "piecewise")).lower()
    t = t.float().clamp(0.0, 1.0)
    if name in {"none", "uniform"}:
        return torch.ones_like(t)
    if name == "piecewise":
        bins = tw.get("bins", None)
        if bins is None:
            bins = [
                [0.00, 0.05, 0.25],
                [0.05, 0.35, 1.00],
                [0.35, 0.65, 0.50],
                [0.65, 1.00, 0.05],
            ]
        weight = torch.zeros_like(t)
        for i, item in enumerate(bins):
            lo, hi, val = float(item[0]), float(item[1]), float(item[2])
            if i == len(bins) - 1:
                mask = (t >= lo) & (t <= hi)
            else:
                mask = (t >= lo) & (t < hi)
            weight = torch.where(mask, torch.full_like(weight, val), weight)
        return weight
    if name == "smooth":
        t_min = float(tw.get("t_min", 0.05))
        t_max = float(tw.get("t_max", 0.65))
        power = float(tw.get("power", 1.0))
        tau = float(tw.get("tau", 0.05))
        clean = (1.0 - t).clamp(0.0, 1.0).pow(power)
        low = torch.sigmoid((t - t_min) / max(tau, 1e-6))
        high = torch.sigmoid((t_max - t) / max(tau, 1e-6))
        w = clean * low * high
        normalize = str(tw.get("normalize", "mean")).lower()
        if normalize == "mean":
            w = w / w.mean().clamp_min(1e-8)
        return w
    raise ValueError(f"Unknown adversarial.t_weight.name={name!r}")


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, *, loss: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss == "hinge":
        real = F.relu(1.0 - real_logits)
        fake = F.relu(1.0 + fake_logits)
    elif loss in {"logistic", "nonsat", "non_saturating"}:
        real = F.softplus(-real_logits)
        fake = F.softplus(fake_logits)
    else:
        raise ValueError(f"Unknown adversarial loss={loss!r}")
    return real + fake, real, fake


def generator_loss(fake_logits: torch.Tensor, *, loss: str) -> torch.Tensor:
    if loss == "hinge":
        return -fake_logits
    if loss in {"logistic", "nonsat", "non_saturating"}:
        return F.softplus(-fake_logits)
    raise ValueError(f"Unknown adversarial loss={loss!r}")


def accuracy_metrics(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    real_prob = torch.sigmoid(real_logits)
    fake_prob = torch.sigmoid(fake_logits)
    real_acc = (real_logits > 0).float().mean()
    fake_acc = (fake_logits < 0).float().mean()
    return {
        "d_real_logit_mean": real_logits.mean(),
        "d_fake_logit_mean": fake_logits.mean(),
        "d_real_prob_mean": real_prob.mean(),
        "d_fake_prob_mean": fake_prob.mean(),
        "d_acc_real": real_acc,
        "d_acc_fake": fake_acc,
        "d_acc_total": 0.5 * (real_acc + fake_acc),
    }


def r1_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.square().view(grad.shape[0], -1).sum(dim=1)
