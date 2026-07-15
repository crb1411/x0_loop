from __future__ import annotations

from dataclasses import dataclass

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch as ProcessForwardBatch
from x0loop.core.time_sampling import TimeSampler, build_time_sampler
from x0loop.losses.atomic import CompositeLoss, match_formula
from x0loop.models.denoiser import Denoiser


@dataclass(frozen=True)
class ConsistencyTrainingConfig:
    enabled: bool = False
    consistent_weight: float = 0.0
    target: str = "v"
    order_times: bool = True


def build_consistency_training_config(cfg: dict) -> ConsistencyTrainingConfig:
    raw = cfg.get("consistency_training", {}) or {}
    consistent_weight = float(raw.get("consistent_weight", raw.get("consistent_weights", raw.get("weight", 0.0))))
    if consistent_weight < 0.0:
        raise ValueError(f"consistency_training.consistent_weight must be >= 0, got {consistent_weight}")
    enabled = bool(raw.get("enabled", consistent_weight > 0.0))
    target = str(raw.get("target", "v")).lower()
    if target not in {"v"}:
        raise ValueError(f"consistency_training.target currently supports only 'v', got {target!r}")
    return ConsistencyTrainingConfig(
        enabled=enabled,
        consistent_weight=consistent_weight,
        target=target,
        order_times=bool(raw.get("order_times", True)),
    )


def build_consistency_time_sampler(cfg: dict, schedule, default_sampler: TimeSampler) -> TimeSampler:
    raw = cfg.get("consistency_training", {}) or {}
    sampler_cfg = raw.get("time_sampler", None)
    if sampler_cfg is None:
        return default_sampler
    return build_time_sampler({"time_sampler": sampler_cfg}, schedule)


def should_add_consistency_loss(cfg: ConsistencyTrainingConfig) -> bool:
    return cfg.enabled and cfg.consistent_weight > 0.0


def _sample_times(
    sampler: TimeSampler,
    n: int,
    *,
    device: torch.device,
    order_times: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_a = sampler.sample(n, device=device).float()
    t_b = sampler.sample(n, device=device).float()
    if not order_times:
        return t_a, t_b
    return torch.maximum(t_a, t_b), torch.minimum(t_a, t_b)


def _path_sample(process: BaseProcess, x0: torch.Tensor, endpoint: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    alpha = process.schedule.alpha(t)
    sigma = process.schedule.sigma(t)
    a = process._reshape_coeff(alpha, x0)
    s = process._reshape_coeff(sigma, x0)
    return a * x0 + s * endpoint


def _v_prediction(process: BaseProcess, x: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return process.v_from_output(x, t, out, aux={})


def consistency_loss_dict(
    *,
    process: BaseProcess,
    loss_fn: CompositeLoss,
    fb_t: ProcessForwardBatch,
    t_weight: torch.Tensor,
    x_r: torch.Tensor,
    out_t: torch.Tensor,
    out_r: torch.Tensor,
) -> dict[str, torch.Tensor]:
    atoms = [atom for atom in loss_fn.atoms if atom.target == "v"]
    if not atoms:
        raise ValueError("consistency_training requires at least one loss term with target: v")

    pred_t = _v_prediction(process, fb_t.xt, fb_t.t, out_t)
    pred_r = _v_prediction(process, x_r, fb_t.t, out_r)
    contributions: list[torch.Tensor] = []
    raw_values: list[torch.Tensor] = []
    fb_weight = ProcessForwardBatch(x0=fb_t.x0, t=t_weight, xt=fb_t.xt, endpoint=fb_t.endpoint)
    for atom in atoms:
        raw = match_formula(
            atom.formula,
            pred_t,
            pred_r,
            delta=atom.delta,
            block_size=atom.block_size,
            temperature=atom.temperature,
            eps=atom.eps,
            channel_reduce=atom.channel_reduce,
        )
        raw_values.append(raw)
        contributions.append(atom.weight_term(raw, fb_weight))

    inner = sum(contributions)
    outer_weight = loss_fn.outer_weight(fb_weight, inner)
    weighted = outer_weight * inner
    raw_mean = sum(raw_values).mean()
    return {
        "total": weighted.mean(),
        "loss_weighted": weighted.mean(),
        "loss_no_weight": inner.mean(),
        "weight": outer_weight.mean(),
        "loss_v": raw_mean,
        "loss_v_mse": raw_mean,
    }


def compute_consistency_training_batch(
    *,
    cfg: ConsistencyTrainingConfig,
    sampler: TimeSampler,
    denoiser: Denoiser,
    process: BaseProcess,
    loss_fn: CompositeLoss,
    x0: torch.Tensor,
    cond: torch.Tensor | None,
) -> tuple[ProcessForwardBatch, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    bsz = x0.shape[0]
    t, r = _sample_times(sampler, bsz, device=x0.device, order_times=cfg.order_times)
    t_weight = torch.maximum(t, r)
    fb_t = process.forward_sample(x0=x0, t=t)
    x_t = fb_t.xt
    x_r = _path_sample(process, x0, fb_t.endpoint, r)

    t_model = denoiser.training_time_condition(t)
    model_x = torch.cat([x_t, x_r], dim=0)
    model_t = torch.cat([t_model, t_model], dim=0)
    model_cond = torch.cat([cond, cond], dim=0) if cond is not None else None
    model_out = denoiser.forward(model_x, model_t, cond=model_cond)
    out_t, out_r = model_out.chunk(2, dim=0)
    loss_dict = consistency_loss_dict(
        process=process,
        loss_fn=loss_fn,
        fb_t=fb_t,
        t_weight=t_weight,
        x_r=x_r,
        out_t=out_t,
        out_r=out_r,
    )
    return fb_t, out_t, x_r, r, loss_dict
