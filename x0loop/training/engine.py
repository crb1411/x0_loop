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
from x0loop.training.clean_loop import CleanLoopBank, CleanLoopConfig, build_clean_loop_config
from x0loop.training.context import ForwardBatch as TrainForwardBatch
from x0loop.training.context import LoopConfig, ModelContext, ResumeState, RuntimeContext
from x0loop.training.factories import build_augment, build_data_context, build_discriminator, build_model_context, build_process, build_schedule, init_runtime, load_resume_state
from x0loop.training.metrics import TimeBinAccumulator, endpoint_loss_label
from x0loop.training.optimization import amp_dtype_for_precision, build_step_lr_schedule, maybe_make_scaler
from x0loop.training.evaluation import run_eval_if_due
from x0loop.training.generative_eval import run_final_generative_eval, run_generative_eval_if_due
from x0loop.training.checkpointing import save_checkpoint_if_due, save_final_checkpoint
from x0loop.training.sampling import apply_classifier_free_label_dropout, run_mudata_observation_if_due, run_sampling_if_due
from x0loop.utils.ema import EMA
from x0loop.utils.fsdp import clip_grad_norm
from x0loop.utils.logger import Logger, MetricLogger


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    seconds_i = max(0, int(round(float(seconds))))
    days, rem = divmod(seconds_i, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return "".join(parts)


class ProgressEstimator:
    def __init__(self, cfg: dict, loop_cfg: LoopConfig, *, start_step: int):
        self.cfg = cfg
        self.loop_cfg = loop_cfg
        self.start_step = int(start_step)
        self.start_time = time.time()
        self.gen_eval_durations: list[float] = []

    def record_gen_eval(self, duration_s: float | None) -> None:
        if duration_s is not None and duration_s > 0 and math.isfinite(float(duration_s)):
            self.gen_eval_durations.append(float(duration_s))

    def _future_periodic_gen_eval_count(self, step: int) -> int:
        gen_cfg = self.cfg.get("gen_eval", {}) or {}
        if not bool(gen_cfg.get("enabled", False)):
            return 0
        every_steps = int(gen_cfg.get("every_steps", 10000))
        if every_steps <= 0:
            return 0
        next_step = ((int(step) // every_steps) + 1) * every_steps
        if next_step > self.loop_cfg.total_steps:
            return 0
        return ((self.loop_cfg.total_steps - next_step) // every_steps) + 1

    def _final_gen_eval_scale(self) -> float:
        gen_cfg = self.cfg.get("gen_eval", {}) or {}
        final_cfg = gen_cfg.get("final", {}) or {}
        if not bool(gen_cfg.get("enabled", False)) or not bool(final_cfg.get("enabled", gen_cfg.get("final_enabled", True))):
            return 0.0
        base_work = int(gen_cfg.get("num_samples", gen_cfg.get("num", 5000))) * int(gen_cfg.get("steps", 20))
        final_work = int(final_cfg.get("num_samples", gen_cfg.get("final_num_samples", 20000))) * int(final_cfg.get("steps", gen_cfg.get("final_steps", 50)))
        if base_work <= 0 or final_work <= 0:
            return 0.0
        return float(final_work) / float(base_work)

    def metrics(self, *, step: int, train_step_s: float | None) -> dict[str, float | int | str]:
        now = time.time()
        elapsed_s = max(0.0, now - self.start_time)
        completed_since_start = max(1, int(step) - self.start_step)
        fallback_step_s = elapsed_s / float(completed_since_start)
        avg_train_step_s = float(train_step_s) if train_step_s is not None and float(train_step_s) > 0 else fallback_step_s
        remaining_steps = max(0, self.loop_cfg.total_steps - int(step))
        eta_train_s = remaining_steps * avg_train_step_s

        avg_gen_eval_s = sum(self.gen_eval_durations) / len(self.gen_eval_durations) if self.gen_eval_durations else 0.0
        eta_periodic_gen_s = self._future_periodic_gen_eval_count(step) * avg_gen_eval_s
        eta_final_gen_s = avg_gen_eval_s * self._final_gen_eval_scale() if self.gen_eval_durations else 0.0
        eta_geneval_s = eta_periodic_gen_s + eta_final_gen_s
        eta_total_s = eta_train_s + eta_geneval_s
        total_est_s = elapsed_s + eta_total_s

        return {
            "progress_pct": 100.0 * float(min(max(int(step), 0), self.loop_cfg.total_steps)) / max(1, self.loop_cfg.total_steps),
            "elapsed_s": elapsed_s,
            "eta_train_s": eta_train_s,
            "eta_geneval_s": eta_geneval_s,
            "eta_total_s": eta_total_s,
            "total_est_s": total_est_s,
            "elapsed": _format_duration(elapsed_s),
            "eta_train": _format_duration(eta_train_s),
            "eta_geneval": _format_duration(eta_geneval_s) if self.gen_eval_durations else "n/a",
            "eta_total": _format_duration(eta_total_s),
            "total_est": _format_duration(total_est_s),
            "gen_eval_observations": len(self.gen_eval_durations),
            "avg_gen_eval_s": avg_gen_eval_s,
        }


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
    for line in _format_lr_shape(loop_cfg):
        logger.log_text(line)


_SHAPE_POINTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
_BAR_WIDTH = 40


def _as_vector(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 1:
        x = x.view(x.shape[0], -1).mean(dim=1)
    return x.detach().float().cpu()


def _bar(value: float, max_value: float, *, width: int = _BAR_WIDTH) -> str:
    if max_value <= 0.0 or not math.isfinite(max_value):
        n = 0
    else:
        n = int(round(width * max(0.0, float(value)) / max_value))
    return "█" * max(0, min(width, n))


def _format_bar_shape(prefix: str, label: str, rows: list[tuple[float, float]], *, x_name: str, value_fmt: str = ".3f") -> list[str]:
    if not rows:
        return []
    values = [float(v) for _, v in rows]
    max_value = max(values) if values else 0.0
    mean_value = sum(values) / len(values) if values else 0.0
    lines = [
        f"[{prefix}] {label} shape | mean={mean_value:.4g} min={min(values):.4g} max={max_value:.4g}"
    ]
    for x, value in rows:
        lines.append(f"[{prefix}] {x_name}={x:>4.2f}  {value:{value_fmt}}  {_bar(value, max_value)}")
    return lines


def _format_loss_weight_shape(loss_fn) -> list[str]:
    t_points = torch.tensor(_SHAPE_POINTS, dtype=torch.float32)
    t_grid = (torch.arange(2000, dtype=torch.float32) + 0.5) / 2000.0

    def _shape(name: str, fn) -> list[str]:
        point_values = _as_vector(fn(t_points))
        grid_values = _as_vector(fn(t_grid))
        rows = [(float(t), float(v)) for t, v in zip(t_points, point_values, strict=True)]
        lines = _format_bar_shape("loss_weight", name, rows, x_name="t")
        lines[0] = (
            f"[loss_weight] {name} shape | "
            f"mean={float(grid_values.mean()):.4g} min={float(grid_values.min()):.4g} max={float(grid_values.max()):.4g}"
        )
        return lines

    lines = _shape("outer", lambda t: loss_fn.outer_weight(SimpleNamespace(t=t), torch.ones_like(t)))
    for atom_index, atom in enumerate(loss_fn.atoms):
        if atom.weight_fn is None:
            continue
        lines.extend(_shape(f"term{atom_index}:{atom.target}", lambda t, atom=atom: atom.weight_fn(t, None)))
    return lines


def _format_time_sampler_shape(time_sampler) -> list[str]:
    sample_count = 20000
    t = time_sampler.sample(sample_count, device=torch.device("cpu")).detach().float().clamp(0.0, 1.0)
    points = torch.tensor(_SHAPE_POINTS, dtype=torch.float32)
    mids = 0.5 * (points[:-1] + points[1:])
    edges = torch.cat([torch.tensor([0.0]), mids, torch.tensor([1.0])])
    idx = torch.bucketize(t, edges[1:-1], right=False)
    hist = torch.bincount(idx, minlength=edges.numel() - 1).float()
    widths = edges[1:] - edges[:-1]
    density = hist / float(sample_count) / widths.clamp_min(1e-8)
    rows = [(float(point), float(value)) for point, value in zip(points, density, strict=True)]
    return _format_bar_shape("time_sampler", "sample_density", rows, x_name="t")


def _format_lr_shape(loop_cfg: LoopConfig) -> list[str]:
    if loop_cfg.total_steps <= 0:
        return []
    rows: list[tuple[float, float]] = []
    for frac in _SHAPE_POINTS:
        step = min(loop_cfg.total_steps - 1, max(0, int(round(frac * float(loop_cfg.total_steps - 1)))))
        rows.append((float(frac), float(loop_cfg.lr_for_step(step))))
    return _format_bar_shape("lr", "schedule", rows, x_name="p", value_fmt=".4g")


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
    step: int,
    clean_loop_cfg: CleanLoopConfig | None = None,
    clean_loop_bank: CleanLoopBank | None = None,
) -> TrainForwardBatch:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    if cond is not None:
        cond = apply_classifier_free_label_dropout(cond, null_class_id=int(model_ctx.model_cfg.num_classes), drop_prob=float(cfg["train"].get("class_dropout_prob", 0.0)))
    if augment_mode == "data_only":
        x0 = augment.apply(x0, augment.sample_params(bsz, device=runtime.device))

    clean_enabled = clean_loop_cfg is not None and clean_loop_cfg.enabled and clean_loop_bank is not None
    requested_bank = int(bsz * clean_loop_cfg.bank_prob) if clean_enabled else 0
    n_bank = 0
    if clean_enabled and step >= clean_loop_cfg.warmup_steps and requested_bank > 0:
        n_bank = min(requested_bank, len(clean_loop_bank))
    n_fresh = bsz - n_bank
    if n_fresh <= 0:
        raise ValueError("clean_loop requires at least one fresh sample per batch; reduce clean_loop.bank_prob.")

    fresh_x0 = x0[:n_fresh]
    fresh_cond = cond[:n_fresh] if cond is not None else None
    bank_x = bank_cond = bank_x0 = bank_steps = None
    if clean_enabled and n_bank > 0:
        bank_x, bank_cond, bank_x0, bank_steps = clean_loop_bank.sample(n_bank, device=runtime.device, dtype=x0.dtype)

    with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype_for_precision(model_ctx.precision), enabled=(model_ctx.precision in {"bf16", "fp16"})):
        if denoiser.loss_fn is None:
            raise ValueError("Denoiser requires loss_fn for training.")
        fresh_fb = denoiser.make_forward_batch(fresh_x0)
        fresh_t_model = denoiser.training_time_condition(fresh_fb.t)
        model_x = fresh_fb.xt
        model_t = fresh_t_model
        model_cond = fresh_cond
        bank_t = None
        if bank_x is not None:
            bank_t = torch.full((bank_x.shape[0],), clean_loop_cfg.time_constant, device=runtime.device, dtype=fresh_fb.t.dtype)
            model_x = torch.cat([model_x, bank_x], dim=0)
            model_t = torch.cat([model_t, bank_t], dim=0)
            if fresh_cond is not None:
                if bank_cond is None:
                    raise ValueError("clean_loop bank was created without labels but current training uses labels.")
                model_cond = torch.cat([fresh_cond, bank_cond], dim=0)

        model_out = denoiser.forward(model_x, model_t, cond=model_cond)
        fresh_out = model_out[:n_fresh]
        loss_dict = denoiser.loss_fn(process, fresh_fb, fresh_out)
        bank_out = model_out[n_fresh:] if bank_x is not None else None
        bank_loss = None
        bank_pred_x0 = None
        if bank_x is not None and bank_t is not None and bank_out is not None and bank_x0 is not None:
            bank_pred_x0 = process.x0_from_output(bank_x, bank_t, bank_out, aux={})
            bank_per_example = regress("mse", bank_pred_x0, bank_x0)
            bank_loss = bank_per_example.mean()

        with torch.no_grad():
            unweighted = _diagnostic_losses(process, fresh_fb, fresh_out)
            fresh_x0_hat = process.x0_from_output(fresh_fb.xt, fresh_fb.t, fresh_out.detach(), aux={})

    extra_metrics = {k: v for k, v in loss_dict.items() if k != "total"}
    extra_metrics.update({f"fresh/{k}": v for k, v in loss_dict.items() if k != "total"})
    fresh_loss = loss_dict["total"]
    fresh_scale = float(n_fresh) / float(bsz)
    fresh_contrib = fresh_loss * fresh_scale
    total_loss = fresh_contrib
    extra_metrics.update({
        "loss_fresh": fresh_loss.detach(),
        "loss_fresh_contrib": fresh_contrib.detach(),
        "fresh/loss": fresh_loss.detach(),
        "fresh/loss_contrib": fresh_contrib.detach(),
    })

    if clean_enabled:
        if bank_loss is not None and bank_steps is not None and bank_pred_x0 is not None and bank_x0 is not None:
            bank_scale = float(n_bank) / float(bsz)
            bank_contrib = bank_scale * clean_loop_cfg.loss_bank_weight * bank_loss
            total_loss = total_loss + bank_contrib
            extra_metrics.update({
                "clean/loss_bank": bank_loss.detach(),
                "clean/loss_bank_weighted": bank_contrib.detach(),
                "clean/loss_bank_contrib": bank_contrib.detach(),
                "clean/loss_bank_weight": clean_loop_cfg.loss_bank_weight,
                "clean/bank_scale": bank_scale,
                "clean/bank_age": (float(step) - bank_steps.detach().float().mean()).clamp_min(0.0),
            })

        add_x_in = fresh_x0_hat
        add_x0 = fresh_fb.x0
        add_cond = fresh_cond
        if bank_pred_x0 is not None and bank_x0 is not None:
            add_x_in = torch.cat([add_x_in, bank_pred_x0.detach()], dim=0)
            add_x0 = torch.cat([add_x0, bank_x0], dim=0)
            if add_cond is not None:
                if bank_cond is None:
                    raise ValueError("clean_loop bank was created without labels but current training uses labels.")
                add_cond = torch.cat([add_cond, bank_cond], dim=0)
        clean_loop_bank.add(x_in=add_x_in, x0=add_x0, cond=add_cond, step=step)
        extra_metrics.update({
            "clean/bank_size": float(len(clean_loop_bank)),
            "clean/bank_prob": clean_loop_cfg.bank_prob,
            "clean/bank_n": float(n_bank),
            "clean/fresh_n": float(n_fresh),
            "clean/fresh_scale": fresh_scale,
        })
    return TrainForwardBatch(loss=total_loss, loss_by_target=unweighted, batch_size=bsz, cond=fresh_cond, fb=fresh_fb, out=fresh_out, extra_metrics=extra_metrics)


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
        with torch.no_grad():
            metrics.update({
                "gan/d_loss": d_loss.detach(),
                "gan/d_real_loss": _weighted_mean(per_real.detach(), weights),
                "gan/d_fake_loss": _weighted_mean(per_fake.detach(), weights),
                "gan/r1_penalty": r1.detach(),
                "gan/d_lr": float(d_optimizer.param_groups[0]["lr"]),
            })
            for key, val in accuracy_metrics(real_logits.detach(), fake_logits.detach()).items():
                metrics[f"gan/{key}"] = val
            tbin_values.update({
                "drl": per_real.detach(),
                "dfl": per_fake.detach(),
                "dacc": 0.5 * ((real_logits.detach() > 0).float() + (fake_logits.detach() < 0).float()),
            })

    _set_requires_grad(discriminator, False)
    fake_g = process.x0_from_output(fb.xt, fb.t, fwd.out, aux={}).to(dtype=disc_dtype)
    fake_logits_g = discriminator(fake_g, fb.t, fwd.cond)
    per_g_adv = generator_loss(fake_logits_g, loss=adv_cfg.loss)
    g_adv = _weighted_mean(per_g_adv, weights)
    fwd.loss = fwd.loss + float(g_weight) * g_adv
    metrics["gan/g_adv_loss"] = g_adv.detach()
    tbin_values["gadv"] = per_g_adv.detach()
    fwd.extra_metrics = {**(fwd.extra_metrics or {}), **metrics}
    fwd.extra_tbin = {**(fwd.extra_tbin or {}), **tbin_values}
    _set_requires_grad(discriminator, True)


def log_training_step(*, runtime: RuntimeContext, loop_cfg: LoopConfig, resume: ResumeState, meters: MetricLogger, tbin_stats: TimeBinAccumulator, epoch: int, micro_step: int, current_accum_steps: int, progress: ProgressEstimator | None = None) -> None:
    force_log = resume.run_step <= 20
    if not force_log and (resume.global_step % loop_cfg.log_every != 0):
        return
    meters.reduce_distributed()
    kv = meters.get_log_dict()
    kv["epoch"] = epoch
    kv["micro_step"] = micro_step + 1
    kv["accumulation_steps"] = current_accum_steps
    if loop_cfg.grad_clip > 0:
        kv["grad_clip"] = loop_cfg.grad_clip
    if torch.cuda.is_available():
        kv["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device=runtime.device) / (1024**3)
    if progress is not None:
        kv.update(progress.metrics(step=resume.global_step, train_step_s=kv.get("iter_s")))
    kv["summary"] = tbin_stats.summary(is_distributed=runtime.is_distributed)
    runtime.logger.log_kv(resume.global_step, kv, total_steps=loop_cfg.total_steps)
    tbin_stats.reset()


def train(cfg: dict) -> None:
    # Distributed/device setup: ranks, device, seeds, logger, out_dir.
    runtime = init_runtime(cfg)
    # Dataset + dataloaders (train/eval) and the distributed sampler.
    data_ctx = build_data_context(cfg, runtime)
    # The network and its runtime wrappers (DDP/FSDP, precision, EMA target).
    model_ctx = build_model_context(cfg, runtime)
    # alpha(t)/sigma(t) time schedule shared by the process and loss weighting.
    schedule = build_schedule(cfg)
    # How training timesteps t are drawn (here: logit-normal).
    time_sampler = build_time_sampler(cfg, schedule)
    # Forward/reverse process: x_t = alpha x0 + sigma * endpoint, plus the sampler.
    # .to(device) because it may hold learnable parameters (e.g. mu_data endpoint).
    process = build_process(cfg, schedule).to(runtime.device)
    # Composite loss: target spaces (x0/endpoint/v/mudata) x formula x t-weighting.
    loss_fn = build_loss(cfg["loss"], schedule)
    # Optional adversarial head (None unless gan/adversarial is configured).
    discriminator = build_discriminator(cfg, runtime)
    # Bundle net+process+loss into one module; also applies the t-conditioning jitter.
    denoiser = Denoiser(
        model_ctx.model,
        process=process,
        loss_fn=loss_fn,
        time_sampler=time_sampler,
        time_condition_jitter=cfg.get("time_condition_jitter", None),
        model_conditioning=cfg.get("model_conditioning", None),
    )
    clean_loop_cfg = build_clean_loop_config(cfg)
    clean_loop_bank = CleanLoopBank(clean_loop_cfg) if clean_loop_cfg.enabled else None
    # Data augmentation pipeline and where it is applied (data_only here).
    augment, augment_mode = build_augment(cfg)
    if runtime.is_main:
        atom_descs = ", ".join(repr(atom) for atom in loss_fn.atoms)
        runtime.logger.log_text(f"[process] name={cfg.get('process', {}).get('name')} output_target={process.output_target} schedule={schedule.mode}")
        runtime.logger.log_text(f"[loss] {atom_descs}")
        for line in _format_loss_weight_shape(loss_fn):
            runtime.logger.log_text(line)
        runtime.logger.log_text(f"[time_sampler] {cfg.get('time_sampler', {'name': 'legacy'})}")
        for line in _format_time_sampler_shape(time_sampler):
            runtime.logger.log_text(line)
        runtime.logger.log_text(f"[time_condition_jitter] {cfg.get('time_condition_jitter', {'enabled': False})}")
        runtime.logger.log_text(f"[model_conditioning] {cfg.get('model_conditioning', {'ignore_time': False})}")
        runtime.logger.log_text(f"[clean_loop] {cfg.get('clean_loop', {'enabled': False})}")
        runtime.logger.log_text(f"[adversarial] {cfg.get('adversarial', {'enabled': False})}")
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=float(cfg["train"].get("lr", 1e-4)), betas=(0.9, 0.95), weight_decay=float(cfg["train"].get("weight_decay", 0.05)))
    d_optimizer = None
    if discriminator is not None:
        dc = cfg.get("discriminator", {}) or {}
        betas = dc.get("betas", [0.0, 0.99])
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(dc.get("lr", 2e-4)),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(dc.get("weight_decay", 0.0)),
        )
    scaler = maybe_make_scaler(precision=model_ctx.precision, use_fsdp=model_ctx.use_fsdp)
    ema = EMA(model=denoiser, decay=float(cfg["train"].get("ema_decay", 0.9999))) if bool(cfg["train"].get("use_ema", True)) else None
    resume = load_resume_state(cfg, denoiser=denoiser, optimizer=optimizer, scaler=scaler, ema=ema, discriminator=discriminator, d_optimizer=d_optimizer, runtime=runtime)
    meters = MetricLogger(window_size=int(cfg["logging"].get("window_size", 20)))
    loop_cfg = build_loop_config(cfg, data_ctx.loader, runtime.distributed_cfg)
    if runtime.is_main:
        log_loop_config(runtime.logger, loop_cfg)
    progress = ProgressEstimator(cfg, loop_cfg, start_step=resume.global_step)
    tbin_stats = TimeBinAccumulator(num_bins=loop_cfg.tbin_count, device=runtime.device)
    epoch = resume.start_epoch

    def _save_final_checkpoint() -> None:
        final_extra_state = None
        if discriminator is not None:
            final_extra_state = {"discriminator": discriminator.state_dict()}
            if d_optimizer is not None:
                final_extra_state["d_optimizer"] = d_optimizer.state_dict()
        save_final_checkpoint(
            cfg=cfg, denoiser=denoiser, runtime=runtime, optimizer=optimizer, scaler=scaler,
            ema=ema, extra_state=final_extra_state, resume=resume, epoch=epoch,
        )

    try:
        denoiser.train()
        iter_start = time.time()
        for epoch in range(resume.start_epoch, loop_cfg.epochs):
            if data_ctx.sampler is not None:
                data_ctx.sampler.set_epoch(epoch)
            use_label_cond = int(cfg["model"].get("num_classes", 0)) > 0
            for micro_step, (x0, y) in enumerate(data_ctx.loader):
                accum_index = micro_step % loop_cfg.gradient_accumulation_steps
                update_step = accum_index == 0
                remaining_micro_steps = loop_cfg.micro_steps_per_epoch - micro_step
                current_accum_steps = min(loop_cfg.gradient_accumulation_steps, remaining_micro_steps)
                did_optimizer_step = should_step_optimizer(micro_step, loop_cfg)
                if update_step:
                    step_lr = float(loop_cfg.lr_for_step(resume.global_step))
                    for pg in optimizer.param_groups:
                        pg["lr"] = step_lr
                    optimizer.zero_grad(set_to_none=True)
                with maybe_no_sync(model_ctx.model, enabled=model_ctx.use_ddp and not did_optimizer_step):
                    fwd = compute_forward_batch(
                        cfg=cfg,
                        runtime=runtime,
                        model_ctx=model_ctx,
                        denoiser=denoiser,
                        process=process,
                        augment=augment,
                        augment_mode=augment_mode,
                        x0=x0,
                        y=y,
                        use_label_cond=use_label_cond,
                        step=resume.global_step,
                        clean_loop_cfg=clean_loop_cfg,
                        clean_loop_bank=clean_loop_bank,
                    )
                    maybe_apply_adversarial(cfg=cfg, fwd=fwd, process=process, discriminator=discriminator, d_optimizer=d_optimizer, scaler=scaler, step=resume.global_step)
                    backward_loss(fwd.loss, current_accum_steps=current_accum_steps, scaler=scaler)
                grad_norm = None
                if did_optimizer_step:
                    sync_process_grads(process, world_size=runtime.world_size)
                    effective_clip = 0.0 if resume.global_step < 10000 else loop_cfg.grad_clip
                    grad_norm = step_optimizer(denoiser, optimizer, scaler, effective_clip)
                if did_optimizer_step and ema is not None:
                    ema.update(denoiser)
                tbin_stats.update(schedule=schedule, process=process, loss_fn=loss_fn, fb=fwd.fb, out=fwd.out, extra_values=fwd.extra_tbin)
                iter_time = time.time() - iter_start
                iter_start = time.time()
                update_train_meters(meters, fwd, lr=float(optimizer.param_groups[0]["lr"]), iter_time=iter_time, world_size=runtime.world_size, grad_norm=grad_norm if did_optimizer_step else None)
                if not did_optimizer_step:
                    continue
                resume.global_step += 1
                resume.run_step += 1
                log_training_step(runtime=runtime, loop_cfg=loop_cfg, resume=resume, meters=meters, tbin_stats=tbin_stats, epoch=epoch, micro_step=micro_step, current_accum_steps=current_accum_steps, progress=progress)
                run_eval_if_due(cfg=cfg, model=denoiser, runtime=runtime, model_ctx=model_ctx, schedule=schedule, time_sampler=time_sampler, process=process, loss_fn=loss_fn, data_ctx=data_ctx, resume=resume, use_label_cond=use_label_cond)
                run_sampling_if_due(cfg=cfg, model=denoiser, runtime=runtime, model_ctx=model_ctx, process=process, ema=ema, loop_cfg=loop_cfg, resume=resume, cond=fwd.cond, use_label_cond=use_label_cond)
                run_mudata_observation_if_due(cfg=cfg, runtime=runtime, process=process, resume=resume)
                extra_state = None
                if discriminator is not None:
                    extra_state = {"discriminator": discriminator.state_dict()}
                    if d_optimizer is not None:
                        extra_state["d_optimizer"] = d_optimizer.state_dict()
                save_checkpoint_if_due(cfg=cfg, denoiser=denoiser, runtime=runtime, optimizer=optimizer, scaler=scaler, ema=ema, extra_state=extra_state, loop_cfg=loop_cfg, resume=resume, epoch=epoch)
                gen_eval_duration = run_generative_eval_if_due(cfg=cfg, model=denoiser, runtime=runtime, model_ctx=model_ctx, process=process, ema=ema, resume=resume)
                progress.record_gen_eval(gen_eval_duration)
                if gen_eval_duration is not None and runtime.is_main:
                    runtime.logger.log_text(f"[progress] gen_eval_duration={_format_duration(gen_eval_duration)} ({gen_eval_duration:.3f}s)")
        # Always persist the latest state at end of training, regardless of cadence.
        _save_final_checkpoint()
        final_gen_eval_duration = run_final_generative_eval(cfg=cfg, model=denoiser, runtime=runtime, model_ctx=model_ctx, process=process, ema=ema, resume=resume)
        progress.record_gen_eval(final_gen_eval_duration)
        if final_gen_eval_duration is not None and runtime.is_main:
            runtime.logger.log_text(f"[progress] final_gen_eval_duration={_format_duration(final_gen_eval_duration)} ({final_gen_eval_duration:.3f}s)")
    except KeyboardInterrupt:
        if runtime.is_main:
            runtime.logger.log_text("[checkpoint] KeyboardInterrupt: saving final checkpoint before exit")
        _save_final_checkpoint()
        raise
    finally:
        runtime.logger.close()
