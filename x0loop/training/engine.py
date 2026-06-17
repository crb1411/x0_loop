from __future__ import annotations

import math
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from x0loop.aug.base import BaseAugment
from x0loop.core.process_base import BaseProcess, ForwardBatch as ProcessForwardBatch
from x0loop.core.time_sampling import build_time_sampler
from x0loop.losses.adversarial import accuracy_metrics, adversarial_weight, build_adversarial_config, discriminator_loss, generator_loss, r1_penalty, t_weight
from x0loop.losses.atomic import regress
from x0loop.losses.spec import build_loss
from x0loop.models.denoiser import Denoiser
from x0loop.training.context import ForwardBatch as TrainForwardBatch
from x0loop.training.context import LoopConfig, ModelContext, ResumeState, RuntimeContext
from x0loop.training.factories import build_augment, build_data_context, build_discriminator, build_model_context, build_process, build_schedule, init_runtime, load_resume_state
from x0loop.training.metrics import TimeBinAccumulator, endpoint_loss_label
from x0loop.training.optimization import amp_dtype_for_precision, build_step_lr_schedule, maybe_make_scaler
from x0loop.training.evaluation import run_eval_if_due
from x0loop.training.generative_eval import run_final_generative_eval, run_generative_eval_if_due
from x0loop.training.checkpointing import save_checkpoint_if_due, save_final_checkpoint
from x0loop.training.sampling import apply_classifier_free_label_dropout, run_sampling_if_due
from x0loop.utils.ema import EMA
from x0loop.utils.fsdp import clip_grad_norm
from x0loop.utils.logger import Logger, MetricLogger


def build_loop_config(cfg: dict, loader: DataLoader, distributed_cfg: dict) -> LoopConfig:
    epochs = int(cfg["train"]["epochs"])
    gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"train.gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}")
    micro_steps_per_epoch = len(loader)
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / gradient_accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    lr_for_step, lr_sched_meta = build_step_lr_schedule(cfg["train"], total_steps=total_steps, steps_per_epoch=optimizer_steps_per_epoch)
    grad_clip_cfg = cfg.get("train", {}).get("max_clip_grad", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = cfg.get("train", {}).get("max_grad_norm", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = distributed_cfg.get("grad_clip_norm", 0.0)
    return LoopConfig(epochs=epochs, gradient_accumulation_steps=gradient_accumulation_steps, micro_steps_per_epoch=micro_steps_per_epoch, optimizer_steps_per_epoch=optimizer_steps_per_epoch, total_steps=total_steps, lr_for_step=lr_for_step, lr_sched_meta=lr_sched_meta, grad_clip=float(grad_clip_cfg), log_every=int(cfg["logging"].get("log_every", 50)), sample_every=int(cfg["logging"].get("sample_every", 2000)), save_every=int(distributed_cfg.get("checkpoint", {}).get("every_steps", 2000)), sample_rank0_only=bool(cfg["logging"].get("sample_rank0_only", True)), tbin_count=int(cfg["logging"].get("t_bins", 20)))


def log_loop_config(logger: Logger, loop_cfg: LoopConfig) -> None:
    logger.log_text(f"[train] gradient_accumulation_steps={loop_cfg.gradient_accumulation_steps}, micro_steps_per_epoch={loop_cfg.micro_steps_per_epoch}, optimizer_steps_per_epoch={loop_cfg.optimizer_steps_per_epoch}")
    logger.log_text(f"[train] grad_clip={loop_cfg.grad_clip}")
    meta = loop_cfg.lr_sched_meta
    logger.log_text(f"[train] lr_scheduler={meta.get('name', 'constant')} meta={meta}")


def _format_loss_weight_shape(loss_fn) -> list[str]:
    t_points = torch.tensor([0.001, 0.25, 0.5, 0.75, 0.999], dtype=torch.float32)
    t_grid = (torch.arange(2000, dtype=torch.float32) + 0.5) / 2000.0

    def _as_vector(x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 1:
            x = x.view(x.shape[0], -1).mean(dim=1)
        return x.detach().float().cpu()

    def _fmt_values(values: torch.Tensor) -> str:
        return " ".join(f"{float(v):>9.4g}" for v in values)

    def _line(name: str, fn) -> str:
        point_values = _as_vector(fn(t_points))
        grid_values = _as_vector(fn(t_grid))
        return (
            f"[loss_weight] {name:<18} {_fmt_values(point_values)} "
            f"| mean={float(grid_values.mean()):.4g} min={float(grid_values.min()):.4g} max={float(grid_values.max()):.4g}"
        )

    lines = [f"[loss_weight] {'t':<18} {_fmt_values(t_points)}"]
    lines.append(
        _line(
            "outer",
            lambda t: loss_fn.outer_weight(SimpleNamespace(t=t), torch.ones_like(t)),
        )
    )
    for atom_index, atom in enumerate(loss_fn.atoms):
        if atom.weight_fn is None:
            continue
        lines.append(_line(f"term{atom_index}:{atom.target}", lambda t, atom=atom: atom.weight_fn(t, None)))
    return lines


def _diagnostic_losses(process: BaseProcess, fb: ProcessForwardBatch, out: torch.Tensor) -> dict[str, torch.Tensor]:
    terminal_label = endpoint_loss_label(process)
    diag = {
        terminal_label: regress("mse", process.endpoint_from_output(fb.xt, fb.t, out, aux={}), process.endpoint_target(fb)).mean(),
        "x0": regress("mse", process.x0_from_output(fb.xt, fb.t, out, aux={}), process.x0_target(fb)).mean(),
        "v": regress("mse", process.v_from_output(fb.xt, fb.t, out, aux={}), process.v_target(fb)).mean(),
    }
    if hasattr(process, "mu_data") and getattr(process, "predict_mudata", False):
        diag["mudata"] = regress("mse", process.mudata_from_output(fb.xt, fb.t, out, aux={}), process.mudata_target(fb)).mean()
    return diag


def compute_forward_batch(
    *,
    cfg: dict,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    denoiser: Denoiser,
    process: BaseProcess,
    augment: BaseAugment,
    augment_mode: str,
    x0: torch.Tensor,
    y: object,
    use_label_cond: bool,
) -> TrainForwardBatch:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    if cond is not None:
        cond = apply_classifier_free_label_dropout(cond, null_class_id=int(model_ctx.model_cfg.num_classes), drop_prob=float(cfg["train"].get("class_dropout_prob", 0.0)))
    if augment_mode == "data_only":
        x0 = augment.apply(x0, augment.sample_params(bsz, device=runtime.device))
    with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype_for_precision(model_ctx.precision), enabled=(model_ctx.precision in {"bf16", "fp16"})):
        batch = denoiser.compute_loss(x0, cond=cond)
        with torch.no_grad():
            unweighted = _diagnostic_losses(process, batch.fb, batch.out)
    extra_metrics = {k: v for k, v in batch.loss_dict.items() if k not in {"total"}}
    return TrainForwardBatch(loss=batch.loss_dict["total"], loss_by_target=unweighted, batch_size=bsz, cond=cond, fb=batch.fb, out=batch.out, extra_metrics=extra_metrics)


def backward_loss(loss: torch.Tensor, *, current_accum_steps: int, scaler: torch.amp.GradScaler | None) -> None:
    loss_for_backward = loss / float(current_accum_steps)
    if scaler is not None:
        scaler.scale(loss_for_backward).backward()
    else:
        loss_for_backward.backward()


def should_step_optimizer(micro_step: int, loop_cfg: LoopConfig) -> bool:
    return ((micro_step + 1) % loop_cfg.gradient_accumulation_steps == 0) or (micro_step + 1 == loop_cfg.micro_steps_per_epoch)


def maybe_no_sync(model: torch.nn.Module, *, enabled: bool):
    no_sync = getattr(model, "no_sync", None)
    if enabled and callable(no_sync):
        return no_sync()
    return nullcontext()


def step_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
) -> torch.Tensor:
    if scaler is not None:
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm(model, grad_clip) if grad_clip > 0 else clip_grad_norm(model, float("inf"))
        scaler.step(optimizer)
        scaler.update()
        return grad_norm
    grad_norm = clip_grad_norm(model, grad_clip) if grad_clip > 0 else clip_grad_norm(model, float("inf"))
    optimizer.step()
    return grad_norm


def sync_process_grads(process: BaseProcess, *, world_size: int) -> None:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return
    for param in process.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(float(world_size))


def update_train_meters(
    meters: MetricLogger,
    fwd: TrainForwardBatch,
    *,
    lr: float,
    iter_time: float,
    world_size: int,
    grad_norm: torch.Tensor | float | None = None,
) -> None:
    meters.update(loss=float(fwd.loss.detach().item()), lr=float(lr), iter_s=float(iter_time), img_s=fwd.batch_size * world_size / max(iter_time, 1e-6))
    for target, val in fwd.loss_by_target.items():
        meters.update(**{f"loss_{target}": float(val.detach().item())})
    for key, val in (fwd.extra_metrics or {}).items():
        if isinstance(val, torch.Tensor):
            val = float(val.detach().float().mean().item())
        meters.update(**{key: float(val)})
    if grad_norm is not None:
        grad_norm_value = float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        meters.update(grad_norm=grad_norm_value)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(enabled)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _discriminator_input_dtype(discriminator: torch.nn.Module) -> torch.dtype:
    return next(discriminator.parameters()).dtype


def maybe_apply_adversarial(
    *,
    cfg: dict,
    fwd: TrainForwardBatch,
    process: BaseProcess,
    discriminator: torch.nn.Module | None,
    d_optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
) -> None:
    adv_cfg = build_adversarial_config(cfg)
    g_weight = adversarial_weight(adv_cfg, step)
    if discriminator is None or d_optimizer is None or g_weight <= 0.0:
        return
    if adv_cfg.fake_space != "x0_hat":
        raise ValueError(f"Only adversarial.fake_space=x0_hat is supported, got {adv_cfg.fake_space!r}")
    if adv_cfg.update_every <= 0 or adv_cfg.d_steps <= 0:
        raise ValueError("adversarial.update_every and adversarial.d_steps must be > 0")

    fb = fwd.fb
    disc_dtype = _discriminator_input_dtype(discriminator)
    weights = t_weight(fb.t, cfg)
    enabled_fraction = (weights > 0).float().mean()
    metrics: dict[str, torch.Tensor | float] = {"gan/g_weight": g_weight, "gan/enabled_t_fraction": enabled_fraction}
    tbin_values: dict[str, torch.Tensor] = {
        "advw": weights.detach(),
    }

    if step % adv_cfg.update_every == 0:
        _set_requires_grad(discriminator, True)
        for _ in range(adv_cfg.d_steps):
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_d = process.x0_from_output(fb.xt, fb.t, fwd.out.detach(), aux={})
                if adv_cfg.clamp_fake_for_d:
                    fake_d = fake_d.clamp(-1.0, 1.0)
                fake_d = fake_d.to(dtype=disc_dtype)
            use_r1 = adv_cfg.r1_gamma > 0.0 and adv_cfg.r1_interval > 0 and (step % adv_cfg.r1_interval == 0)
            real_images = fb.x0.detach().to(dtype=disc_dtype).requires_grad_(use_r1)
            real_logits = discriminator(real_images, fb.t, fwd.cond)
            fake_logits = discriminator(fake_d, fb.t, fwd.cond)
            per_d, per_real, per_fake = discriminator_loss(real_logits, fake_logits, loss=adv_cfg.loss)
            d_loss = _weighted_mean(per_d, weights)
            r1 = torch.zeros((), device=d_loss.device, dtype=d_loss.dtype)
            if use_r1:
                r1 = _weighted_mean(r1_penalty(real_logits, real_images), weights)
                d_loss = d_loss + 0.5 * float(adv_cfg.r1_gamma) * float(adv_cfg.r1_interval) * r1
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                d_loss.backward()
                d_optimizer.step()
            metrics.update({"gan/d_loss": d_loss.detach(), "gan/r1": r1.detach(), "gan/real_acc": accuracy_metrics(real_logits).get("acc"), "gan/fake_acc": accuracy_metrics(fake_logits).get("acc")})
            d_optimizer.zero_grad(set_to_none=True)

    _set_requires_grad(discriminator, False)
    if g_weight > 0:
        fake_g = process.x0_from_output(fb.xt, fb.t, fwd.out, aux={})
        if adv_cfg.clamp_fake_for_g:
            fake_g = fake_g.clamp(-1.0, 1.0)
        fake_g = fake_g.to(dtype=disc_dtype)
        fake_logits_g = discriminator(fake_g, fb.t, fwd.cond)
        g_per = generator_loss(fake_logits_g, loss=adv_cfg.loss)
        g_loss = _weighted_mean(g_per, weights) * float(g_weight)
        fwd.loss = fwd.loss + g_loss
        metrics.update({"gan/g_loss": g_loss.detach(), "gan/fake_g_acc": accuracy_metrics(fake_logits_g).get("acc")})

    fwd.extra_metrics = {**(fwd.extra_metrics or {}), **metrics}
    fwd.extra_tbin = {**(fwd.extra_tbin or {}), **tbin_values}
