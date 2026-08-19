from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
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
from x0loop.training.clean_loop import (
    CleanLoopBank,
    CleanLoopConfig,
    TrajectoryBank,
    TrajectoryBatch,
    build_clean_loop_bank_input,
    build_clean_loop_config,
    sample_clean_loop_t1,
)
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


class TrainingIntervalTimer:
    """Measure training time while synchronizing CUDA only at log cadence."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda_events = device.type == "cuda"
        self.start_event: torch.cuda.Event | None = None
        self.start_time = 0.0
        self.micro_steps = 0
        self.images = 0

    def begin(self) -> None:
        self.micro_steps = 0
        self.images = 0
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def record_step(self, *, batch_size: int, world_size: int) -> None:
        self.micro_steps += 1
        self.images += int(batch_size) * int(world_size)

    def flush_into(self, meters: MetricLogger) -> None:
        if self.micro_steps <= 0:
            return
        if self.use_cuda_events:
            if self.start_event is None:
                raise RuntimeError("TrainingIntervalTimer.begin() must be called before flush_into()")
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            elapsed_s = self.start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed_s = time.perf_counter() - self.start_time
        per_step_s = elapsed_s / float(self.micro_steps)
        images_per_s = float(self.images) / max(elapsed_s, 1.0e-9)
        # Preserve the per-micro-step weighting of the previous meter path.
        for _ in range(self.micro_steps):
            meters.update(iter_s=per_step_s, img_s=images_per_s)


@dataclass(frozen=True)
class TerminalAdversarialPrefix:
    """Detached inference state immediately before the final solver interval."""

    x: torch.Tensor
    t_scalar: torch.Tensor
    s_scalar: torch.Tensor
    real: torch.Tensor
    cond: torch.Tensor | None
    null_cond: torch.Tensor | None


def build_loop_config(cfg: dict, loader: DataLoader, distributed_cfg: dict) -> LoopConfig:
    epochs = int(cfg["train"]["epochs"])
    gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"train.gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}")
    micro_steps_per_epoch = len(loader)
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / gradient_accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    max_steps = cfg["train"].get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError(f"train.max_steps must be > 0 when set, got {max_steps}")
        total_steps = min(total_steps, max_steps)
    run_steps = cfg["train"].get("run_steps")
    if run_steps is not None and int(run_steps) <= 0:
        raise ValueError(f"train.run_steps must be > 0 when set, got {run_steps}")
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
    lr_shape = _format_lr_shape(loop_cfg)
    if lr_shape:
        logger.log_text("\n".join(lr_shape))


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
        lines.append(f"  {x_name}={x:>4.2f}  {value:{value_fmt}}  {_bar(value, max_value)}")
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


def _parameter_gradient_norm(loss: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    """Return the L2 norm of a loss gradient over all trainable parameters.

    ``autograd.grad`` leaves ``parameter.grad`` untouched, so this diagnostic
    can set an auxiliary scale before the normal combined backward pass.
    The retained graph is consumed later by ``backward_loss``.
    """

    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        return torch.zeros((), device=loss.device, dtype=torch.float32)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().square().sum()
    return squared_norm.sqrt()


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
    clean_loop_bank: CleanLoopBank | TrajectoryBank | None = None,
    ema: EMA | None = None,
) -> TrainForwardBatch:
    # Terminal adversarial rollout consumes a separate, deterministically
    # forked RNG stream and is generated before any trainable forward. This
    # preserves FRESH t/noise/data equivalence and lets EMA weights be swapped
    # in without mutating parameters already referenced by an autograd graph.
    terminal_prefix = _prepare_terminal_adversarial_prefix(
        cfg=cfg,
        runtime=runtime,
        model_ctx=model_ctx,
        denoiser=denoiser,
        process=process,
        x0=x0,
        y=y,
        use_label_cond=use_label_cond,
        step=step,
        ema=ema,
    )
    if clean_loop_cfg is not None and clean_loop_cfg.enabled and clean_loop_cfg.version == 2:
        if not isinstance(clean_loop_bank, TrajectoryBank):
            raise TypeError("clean_loop v2 requires a TrajectoryBank")
        fwd = compute_forward_batch_v2(
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
            step=step,
            clean_loop_cfg=clean_loop_cfg,
            clean_loop_bank=clean_loop_bank,
            ema=ema,
        )
        return _attach_terminal_adversarial_payload(
            cfg=cfg,
            model_ctx=model_ctx,
            denoiser=denoiser,
            process=process,
            fwd=fwd,
            prefix=terminal_prefix,
        )
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
    bank_x = bank_cond = bank_x0 = bank_t = bank_steps = None
    if clean_enabled and n_bank > 0:
        bank_x, bank_cond, bank_x0, bank_t, bank_steps = clean_loop_bank.sample(n_bank, device=runtime.device, dtype=x0.dtype)

    with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype_for_precision(model_ctx.precision), enabled=(model_ctx.precision in {"bf16", "fp16"})):
        if denoiser.loss_fn is None:
            raise ValueError("Denoiser requires loss_fn for training.")
        fresh_fb = denoiser.make_forward_batch(fresh_x0)
        fresh_t_model = denoiser.training_time_condition(fresh_fb.t)
        model_x = fresh_fb.xt
        model_t = fresh_t_model
        model_cond = fresh_cond
        if bank_x is not None:
            bank_t = bank_t.to(device=runtime.device, dtype=fresh_fb.t.dtype)
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

    fresh_loss = loss_dict["total"]
    fresh_scale = float(n_fresh) / float(bsz)
    fresh_contrib = fresh_loss * fresh_scale
    total_loss = fresh_contrib
    extra_metrics = {
        "fresh/loss": fresh_loss.detach(),
        "fresh/loss_no_weight": loss_dict["loss_no_weight"].detach(),
        "fresh/weight": loss_dict["weight"].detach(),
    }
    aggregate_loss_keys = {f"loss_{atom.target}" for atom in denoiser.loss_fn.atoms}
    for key, value in loss_dict.items():
        if key in {"total", "loss_weighted", "loss_no_weight", "weight"}:
            continue
        if key in aggregate_loss_keys:
            continue
        if key.startswith("loss_"):
            extra_metrics[f"fresh/{key}"] = value.detach()
    if clean_enabled and n_bank > 0:
        extra_metrics["fresh/loss_contrib"] = fresh_contrib.detach()

    if clean_enabled:
        extra_metrics.update({
            "fresh/loss_contrib": fresh_contrib.detach(),
            "clean/loss_bank": 0.0,
            "clean/loss_bank_contrib": 0.0,
            "clean/loss_bank_weight": clean_loop_cfg.loss_bank_weight,
            "clean/bank_scale": float(n_bank) / float(bsz),
            "clean/warmup_left": float(max(0, clean_loop_cfg.warmup_steps - step)),
        })
        if bank_loss is not None and bank_steps is not None and bank_pred_x0 is not None and bank_x0 is not None:
            bank_scale = float(n_bank) / float(bsz)
            bank_contrib = bank_scale * clean_loop_cfg.loss_bank_weight * bank_loss
            total_loss = total_loss + bank_contrib
            extra_metrics.update({
                "clean/loss_bank": bank_loss.detach(),
                "clean/loss_bank_contrib": bank_contrib.detach(),
                "clean/loss_bank_weight": clean_loop_cfg.loss_bank_weight,
                "clean/bank_scale": bank_scale,
                "clean/bank_age": (float(step) - bank_steps.detach().float().mean()).clamp_min(0.0),
            })

        fresh_add_mask = fresh_fb.t > clean_loop_cfg.t_bank
        fresh_add_t1_all = sample_clean_loop_t1(
            cfg=clean_loop_cfg,
            t=fresh_fb.t,
            time_sampler=denoiser.time_sampler,
            device=runtime.device,
        )
        fresh_xt1_hat_all = build_clean_loop_bank_input(
            cfg=clean_loop_cfg,
            process=process,
            xt=fresh_fb.xt,
            t=fresh_fb.t,
            model_out=fresh_out.detach(),
            x0_hat=fresh_x0_hat,
            t1=fresh_add_t1_all,
        )
        add_x_in = fresh_xt1_hat_all[fresh_add_mask]
        add_x0 = fresh_fb.x0[fresh_add_mask]
        add_t = fresh_add_t1_all[fresh_add_mask]
        add_cond = fresh_cond[fresh_add_mask] if fresh_cond is not None else None
        fresh_add_n = int(fresh_add_mask.detach().sum().item())
        bank_add_n = 0
        if bank_pred_x0 is not None and bank_x0 is not None:
            assert bank_x is not None and bank_t is not None and bank_out is not None
            bank_add_t1 = sample_clean_loop_t1(
                cfg=clean_loop_cfg,
                t=bank_t,
                time_sampler=denoiser.time_sampler,
                device=runtime.device,
            )
            bank_xt1_hat = build_clean_loop_bank_input(
                cfg=clean_loop_cfg,
                process=process,
                xt=bank_x,
                t=bank_t,
                model_out=bank_out.detach(),
                x0_hat=bank_pred_x0.detach(),
                t1=bank_add_t1,
            )
            add_x_in = torch.cat([add_x_in, bank_xt1_hat.detach()], dim=0)
            add_x0 = torch.cat([add_x0, bank_x0], dim=0)
            add_t = torch.cat([add_t, bank_add_t1], dim=0)
            bank_add_n = int(bank_pred_x0.shape[0])
            if add_cond is not None:
                if bank_cond is None:
                    raise ValueError("clean_loop bank was created without labels but current training uses labels.")
                add_cond = torch.cat([add_cond, bank_cond], dim=0)
            elif fresh_cond is not None:
                if bank_cond is None:
                    raise ValueError("clean_loop bank was created without labels but current training uses labels.")
                add_cond = bank_cond
        clean_loop_bank.add(x_in=add_x_in, x0=add_x0, t=add_t, cond=add_cond, step=step)
        extra_metrics.update({
            "clean/bank_size": float(len(clean_loop_bank)),
            "clean/bank_prob": clean_loop_cfg.bank_prob,
            "clean/bank_n": float(n_bank),
            "clean/fresh_n": float(n_fresh),
            "clean/fresh_scale": fresh_scale,
            "clean/t_bank": clean_loop_cfg.t_bank,
            "clean/t1": add_t.detach().float().mean() if add_t.numel() > 0 else 0.0,
            "clean/fresh_add_n": float(fresh_add_n),
            "clean/bank_add_n": float(bank_add_n),
        })
    fwd = TrainForwardBatch(
        loss=total_loss,
        loss_by_target=unweighted,
        batch_size=bsz,
        cond=fresh_cond,
        fb=fresh_fb,
        out=fresh_out,
        extra_metrics=extra_metrics,
    )
    return _attach_terminal_adversarial_payload(
        cfg=cfg,
        model_ctx=model_ctx,
        denoiser=denoiser,
        process=process,
        fwd=fwd,
        prefix=terminal_prefix,
    )


@contextmanager
def _ema_teacher(denoiser: Denoiser, ema: EMA | None):
    was_training = denoiser.training
    if ema is not None:
        ema.store(denoiser)
        ema.copy_to(denoiser)
    denoiser.eval()
    try:
        yield
    finally:
        if ema is not None:
            ema.restore(denoiser)
        denoiser.train(was_training)


@torch.no_grad()
def _teacher_velocity(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    out = denoiser.model_output(
        x,
        t,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    return process.velocity_from_output(x, t, out, aux={})


@torch.no_grad()
def _teacher_targets(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = denoiser.model_output(
        x,
        t,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    return (
        process.velocity_from_output(x, t, out, aux={}),
        process.x0_from_output(x, t, out, aux={}),
    )


@torch.no_grad()
def _teacher_heun_step(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    x: torch.Tensor,
    t_scalar: torch.Tensor,
    s_scalar: torch.Tensor,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
    is_last: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    t = t_scalar.to(device=x.device, dtype=torch.float32).expand(x.shape[0])
    velocity, target_x0 = _teacher_targets(
        denoiser=denoiser,
        process=process,
        x=x,
        t=t,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    dt = s_scalar - t_scalar
    if is_last:
        return x + dt * velocity, velocity, target_x0, None
    x_euler = x + dt * velocity
    s = s_scalar.to(device=x.device, dtype=torch.float32).expand(x.shape[0])
    velocity_s = _teacher_velocity(
        denoiser=denoiser,
        process=process,
        x=x_euler,
        t=s,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    return x + dt * 0.5 * (velocity + velocity_s), velocity, target_x0, velocity_s


def _validate_terminal_kernel(cfg: dict, *, steps: int, sampler: str, guidance_scale: float) -> None:
    """Require the training rollout to be the declared FID inference kernel."""

    gen_eval = cfg.get("gen_eval", {}) or {}
    eval_steps = int(gen_eval.get("steps", 20))
    eval_sampler = str(gen_eval.get("sampler", "heun")).lower()
    eval_guidance = float(gen_eval.get("guidance_scale", 1.0))
    guidance_schedule = gen_eval.get("guidance_schedule")
    if (steps, sampler, guidance_scale) != (eval_steps, eval_sampler, eval_guidance):
        raise ValueError(
            "terminal adversarial kernel must match gen_eval exactly: "
            f"terminal=({steps},{sampler},{guidance_scale}) "
            f"gen_eval=({eval_steps},{eval_sampler},{eval_guidance})"
        )
    if guidance_schedule is not None:
        raise ValueError(
            "terminal adversarial v1 requires gen_eval.guidance_schedule=null "
            "so the rollout and final-step CFG are identical"
        )


@torch.no_grad()
def _terminal_heun_prefix_from_root(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    root: torch.Tensor,
    steps: int,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run every Heun interval except the final Euler interval.

    The returned state is exactly the occupancy seen by the last model call in
    ``FlowProcess.sample(..., sampler="heun")``. The function intentionally has
    no trainable graph; Cycle 03 truncates backpropagation at this boundary.
    """

    pairs = process.schedule.iter_pairs(steps, device=root.device)
    if not pairs:
        raise ValueError("terminal adversarial rollout requires at least one solver pair")
    x = root
    for t_scalar, s_scalar in pairs[:-1]:
        x, _, _, _ = _teacher_heun_step(
            denoiser=denoiser,
            process=process,
            x=x,
            t_scalar=t_scalar,
            s_scalar=s_scalar,
            cond=cond,
            null_cond=null_cond,
            guidance_scale=guidance_scale,
            is_last=False,
        )
    final_t, final_s = pairs[-1]
    return x.detach(), final_t, final_s


def _terminal_last_step(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    x: torch.Tensor,
    t_scalar: torch.Tensor,
    s_scalar: torch.Tensor,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable final Euler step used by the actual Heun sampler."""

    t = t_scalar.to(device=x.device, dtype=torch.float32).expand(x.shape[0])
    out = denoiser.model_output(
        x,
        t,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    velocity = process.velocity_from_output(x, t, out, aux={})
    dt = (s_scalar - t_scalar).to(device=x.device, dtype=x.dtype)
    return x + dt * velocity, out


def _prepare_terminal_adversarial_prefix(
    *,
    cfg: dict,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    denoiser: Denoiser,
    process: BaseProcess,
    x0: torch.Tensor,
    y: object,
    use_label_cond: bool,
    step: int,
    ema: EMA | None,
) -> TerminalAdversarialPrefix | None:
    adv_cfg = build_adversarial_config(cfg)
    if adv_cfg.fake_space != "terminal_x0" or adversarial_weight(adv_cfg, step) <= 0.0:
        return None
    if ema is None:
        raise ValueError("terminal adversarial training requires train.use_ema=true")
    if bool((cfg.get("clean_loop", {}) or {}).get("enabled", False)):
        raise ValueError(
            "Cycle 03 terminal adversarial training cannot be combined with clean_loop; "
            "the registered experiment isolates terminal distribution matching"
        )
    _validate_terminal_kernel(
        cfg,
        steps=adv_cfg.terminal_steps,
        sampler=adv_cfg.terminal_sampler,
        guidance_scale=adv_cfg.terminal_guidance_scale,
    )

    real_all = x0.to(runtime.device, non_blocking=True)
    n = max(1, int(round(real_all.shape[0] * adv_cfg.batch_ratio)))
    real = real_all[:n].detach()
    if use_label_cond:
        if not isinstance(y, torch.Tensor):
            raise ValueError("terminal class-conditional GAN requires tensor labels")
        cond = y[:n].to(runtime.device, non_blocking=True).long()
        null_cond = torch.full_like(cond, int(model_ctx.model_cfg.num_classes))
    else:
        cond = null_cond = None

    amp_enabled = model_ctx.precision in {"bf16", "fp16"}
    rng_devices = [runtime.device.index or 0] if runtime.device.type == "cuda" else []
    rollout_seed = (
        int(cfg["train"].get("seed", 0))
        + 2_000_003
        + int(step) * runtime.world_size
        + runtime.rank
    )
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(rollout_seed)
        root = process.prior_sample(
            (n, *real.shape[1:]),
            device=runtime.device,
            dtype=real.dtype,
        )
        with _ema_teacher(denoiser, ema), torch.autocast(
            device_type=runtime.device.type,
            dtype=amp_dtype_for_precision(model_ctx.precision),
            enabled=amp_enabled,
        ):
            x, t_scalar, s_scalar = _terminal_heun_prefix_from_root(
                denoiser=denoiser,
                process=process,
                root=root,
                steps=adv_cfg.terminal_steps,
                cond=cond,
                null_cond=null_cond,
                guidance_scale=adv_cfg.terminal_guidance_scale,
            )
    return TerminalAdversarialPrefix(
        x=x,
        t_scalar=t_scalar,
        s_scalar=s_scalar,
        real=real,
        cond=cond,
        null_cond=null_cond,
    )


def _attach_terminal_adversarial_payload(
    *,
    cfg: dict,
    model_ctx: ModelContext,
    denoiser: Denoiser,
    process: BaseProcess,
    fwd: TrainForwardBatch,
    prefix: TerminalAdversarialPrefix | None,
) -> TrainForwardBatch:
    if prefix is None:
        return fwd
    adv_cfg = build_adversarial_config(cfg)
    amp_enabled = model_ctx.precision in {"bf16", "fp16"}
    with torch.autocast(
        device_type=prefix.x.device.type,
        dtype=amp_dtype_for_precision(model_ctx.precision),
        enabled=amp_enabled,
    ):
        fake, output = _terminal_last_step(
            denoiser=denoiser,
            process=process,
            x=prefix.x,
            t_scalar=prefix.t_scalar,
            s_scalar=prefix.s_scalar,
            cond=prefix.cond,
            null_cond=prefix.null_cond,
            guidance_scale=adv_cfg.terminal_guidance_scale,
        )
    fwd.adv_real = prefix.real
    fwd.adv_fake = fake
    fwd.adv_cond = prefix.cond
    fwd.adv_t = torch.zeros(fake.shape[0], device=fake.device, dtype=torch.float32)
    fwd.adv_output = output
    fwd.extra_metrics = {
        **(fwd.extra_metrics or {}),
        "gan/terminal_batch": float(fake.shape[0]),
        "gan/terminal_prefix_t": prefix.t_scalar.detach().float(),
        "gan/terminal_fake_rms": fake.detach().float().square().mean().sqrt(),
    }
    return fwd


@torch.no_grad()
def _teacher_heun_step_batched(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    x: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one Heun step when every sample may use a different grid pair.

    A transformer can condition a batch on heterogeneous times. Keeping the
    samples together avoids issuing one tiny teacher forward per occupied
    solver level, which leaves the GPU mostly idle during bank refresh.
    """
    if t.shape != (x.shape[0],) or s.shape != (x.shape[0],):
        raise ValueError("batched Heun times must have shape [batch]")
    velocity = _teacher_velocity(
        denoiser=denoiser,
        process=process,
        x=x,
        t=t,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    dt = (s - t).reshape(-1, *([1] * (x.ndim - 1)))
    x_euler = x + dt * velocity
    velocity_s = _teacher_velocity(
        denoiser=denoiser,
        process=process,
        x=x_euler,
        t=s,
        cond=cond,
        null_cond=null_cond,
        guidance_scale=guidance_scale,
    )
    return x + dt * 0.5 * (velocity + velocity_s), velocity, velocity_s


def _trajectory_batch(
    *,
    x: torch.Tensor,
    target_v: torch.Tensor,
    target_x0: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    solver_index: torch.Tensor,
    root_noise_id: torch.Tensor,
    producer_step: int,
) -> TrajectoryBatch:
    return TrajectoryBatch(
        x=x,
        target_v=target_v,
        target_x0=target_x0,
        t=t,
        cond=cond,
        solver_index=solver_index,
        depth=solver_index.clone(),
        root_noise_id=root_noise_id,
        producer_step=torch.full_like(solver_index, int(producer_step)),
    )


@torch.no_grad()
def _online_trajectory_batch(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    clean_cfg: CleanLoopConfig,
    n: int,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    root_ids: torch.Tensor,
    step: int,
) -> TrajectoryBatch:
    pairs = process.schedule.iter_pairs(clean_cfg.solver_steps, device=device)
    wanted = torch.randint(len(pairs), (n,), device=device)
    x = process.prior_sample((n, *shape), device=device, dtype=dtype)
    selected_x = torch.empty_like(x)
    selected_v = torch.empty_like(x)
    selected_x0 = torch.empty_like(x)
    selected_t = torch.empty((n,), device=device, dtype=torch.float32)
    # Resolve the final occupied level once. The former ``bool(active.any())``
    # synchronized CUDA on every solver level.
    max_wanted = int(wanted.max().item())
    for index, (t_scalar, s_scalar) in enumerate(pairs[:max_wanted + 1]):
        active = wanted >= index
        active_cond = cond[active] if cond is not None else None
        active_null = null_cond[active] if null_cond is not None else None
        next_x, velocity, target_x0, _ = _teacher_heun_step(
            denoiser=denoiser,
            process=process,
            x=x[active],
            t_scalar=t_scalar,
            s_scalar=s_scalar,
            cond=active_cond,
            null_cond=active_null,
            guidance_scale=clean_cfg.guidance_scale,
            is_last=index == len(pairs) - 1,
        )
        chosen = wanted[active] == index
        active_positions = active.nonzero(as_tuple=False).flatten()
        chosen_positions = active_positions[chosen]
        selected_x[chosen_positions] = x[active][chosen]
        selected_v[chosen_positions] = velocity[chosen]
        selected_x0[chosen_positions] = target_x0[chosen].to(dtype=selected_x0.dtype)
        selected_t[chosen_positions] = t_scalar
        x[active] = next_x
    return _trajectory_batch(
        x=selected_x,
        target_v=selected_v,
        target_x0=selected_x0,
        t=selected_t,
        cond=cond,
        solver_index=wanted,
        root_noise_id=root_ids,
        producer_step=step,
    )


@torch.no_grad()
def _refresh_trajectory_bank(
    *,
    denoiser: Denoiser,
    process: BaseProcess,
    clean_cfg: CleanLoopConfig,
    bank: TrajectoryBank,
    n: int,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    cond: torch.Tensor | None,
    null_cond: torch.Tensor | None,
    step: int,
) -> None:
    pairs = process.schedule.iter_pairs(clean_cfg.solver_steps, device=device)
    n_root = max(1, int(round(n * clean_cfg.root_fraction)))
    root_cond = cond[:n_root] if cond is not None else None
    root_null = null_cond[:n_root] if null_cond is not None else None
    root_x = process.prior_sample((n_root, *shape), device=device, dtype=dtype)
    root_t = pairs[0][0].to(device=device, dtype=torch.float32).expand(n_root)
    root_v, root_x0 = _teacher_targets(
        denoiser=denoiser,
        process=process,
        x=root_x,
        t=root_t,
        cond=root_cond,
        null_cond=root_null,
        guidance_scale=clean_cfg.guidance_scale,
    )
    bank.add(_trajectory_batch(
        x=root_x,
        target_v=root_v,
        target_x0=root_x0,
        t=root_t,
        cond=root_cond,
        solver_index=torch.zeros(n_root, device=device, dtype=torch.long),
        root_noise_id=bank.new_root_ids(n_root, device=device),
        producer_step=step,
    ))

    n_advance = max(0, n - n_root)
    if n_advance == 0 or len(bank) == 0:
        return
    parents = bank.sample(n_advance, device=device, dtype=dtype)
    valid = parents.solver_index < len(pairs) - 1
    if not bool(valid.any()):
        return

    parent_x = parents.x[valid]
    parent_index = parents.solver_index[valid]
    parent_cond = parents.cond[valid] if parents.cond is not None else None
    parent_null = null_cond[:1].expand_as(parent_cond) if parent_cond is not None and null_cond is not None else None
    grid_t = torch.stack([pair[0] for pair in pairs]).to(device=device, dtype=torch.float32)
    grid_s = torch.stack([pair[1] for pair in pairs]).to(device=device, dtype=torch.float32)
    parent_t = grid_t[parent_index]
    next_t = grid_s[parent_index]
    next_x, _, _ = _teacher_heun_step_batched(
        denoiser=denoiser,
        process=process,
        x=parent_x,
        t=parent_t,
        s=next_t,
        cond=parent_cond,
        null_cond=parent_null,
        guidance_scale=clean_cfg.guidance_scale,
    )
    # The correct next-grid target is evaluated at the accepted Heun state,
    # not at the Euler predictor used for the corrector.
    next_v, next_x0 = _teacher_targets(
        denoiser=denoiser,
        process=process,
        x=next_x,
        t=next_t,
        cond=parent_cond,
        null_cond=parent_null,
        guidance_scale=clean_cfg.guidance_scale,
    )
    next_index = parent_index + 1
    bank.add(_trajectory_batch(
        x=next_x,
        target_v=next_v,
        target_x0=next_x0,
        t=next_t,
        cond=parent_cond,
        solver_index=next_index,
        root_noise_id=parents.root_noise_id[valid],
        producer_step=step,
    ))


def compute_forward_batch_v2(
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
    clean_loop_cfg: CleanLoopConfig,
    clean_loop_bank: TrajectoryBank,
    ema: EMA | None,
) -> TrainForwardBatch:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    raw_cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    if clean_loop_cfg.mode == "drop":
        n_fresh = max(1, int(round(bsz * (1.0 - clean_loop_cfg.drop_fraction))))
    else:
        n_fresh = bsz
    fresh_x0 = x0[:n_fresh]
    fresh_cond = raw_cond[:n_fresh] if raw_cond is not None else None
    if fresh_cond is not None:
        fresh_cond = apply_classifier_free_label_dropout(
            fresh_cond,
            null_class_id=int(model_ctx.model_cfg.num_classes),
            drop_prob=float(cfg["train"].get("class_dropout_prob", 0.0)),
        )
    if augment_mode == "data_only":
        x0 = augment.apply(x0, augment.sample_params(bsz, device=runtime.device))
        fresh_x0 = x0[:n_fresh]

    # Draw the normal training state before any rollout RNG is consumed. This
    # keeps fresh t/noise aligned between FRESH, BANK-FIX, and ONLINE branches.
    fresh_fb = denoiser.make_forward_batch(fresh_x0)

    aux = None
    bank_refresh_n = 0
    n_aux = max(1, int(round(bsz * clean_loop_cfg.aux_batch_ratio)))
    amp_enabled = model_ctx.precision in {"bf16", "fp16"}
    if clean_loop_cfg.mode in {"bank_fix", "online"} and step >= clean_loop_cfg.warmup_steps:
        aux_cond = raw_cond[:n_aux] if raw_cond is not None else None
        null_cond = torch.full_like(aux_cond, int(model_ctx.model_cfg.num_classes)) if aux_cond is not None else None
        rng_devices = list(range(torch.cuda.device_count())) if runtime.device.type == "cuda" else []
        rollout_seed = int(cfg["train"].get("seed", 0)) + 1_000_003 + step * runtime.world_size + runtime.rank
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(rollout_seed)
            if clean_loop_cfg.mode == "online":
                with _ema_teacher(denoiser, ema), torch.autocast(
                    device_type=runtime.device.type,
                    dtype=amp_dtype_for_precision(model_ctx.precision),
                    enabled=amp_enabled,
                ):
                    aux = _online_trajectory_batch(
                        denoiser=denoiser,
                        process=process,
                        clean_cfg=clean_loop_cfg,
                        n=n_aux,
                        shape=tuple(x0.shape[1:]),
                        device=runtime.device,
                        dtype=x0.dtype,
                        cond=aux_cond,
                        null_cond=null_cond,
                        root_ids=clean_loop_bank.new_root_ids(n_aux, device=runtime.device),
                        step=step,
                    )
            else:
                refresh_now = (
                    len(clean_loop_bank) < n_aux
                    or (step - clean_loop_cfg.warmup_steps) % clean_loop_cfg.refresh_interval == 0
                )
                if refresh_now:
                    bank_refresh_n = n_aux * clean_loop_cfg.refresh_interval
                    if raw_cond is not None:
                        refresh_indices = torch.arange(bank_refresh_n, device=runtime.device) % raw_cond.shape[0]
                        refresh_cond = raw_cond[refresh_indices]
                        refresh_null = torch.full_like(refresh_cond, int(model_ctx.model_cfg.num_classes))
                    else:
                        refresh_cond = refresh_null = None
                    with _ema_teacher(denoiser, ema), torch.autocast(
                        device_type=runtime.device.type,
                        dtype=amp_dtype_for_precision(model_ctx.precision),
                        enabled=amp_enabled,
                    ):
                        _refresh_trajectory_bank(
                            denoiser=denoiser,
                            process=process,
                            clean_cfg=clean_loop_cfg,
                            bank=clean_loop_bank,
                            n=bank_refresh_n,
                            shape=tuple(x0.shape[1:]),
                            device=runtime.device,
                            dtype=x0.dtype,
                            cond=refresh_cond,
                            null_cond=refresh_null,
                            step=step,
                        )
                aux = clean_loop_bank.sample(n_aux, device=runtime.device, dtype=x0.dtype)

    with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype_for_precision(model_ctx.precision), enabled=amp_enabled):
        if denoiser.loss_fn is None:
            raise ValueError("Denoiser requires loss_fn for training.")
        fresh_out = denoiser.forward(fresh_fb.xt, denoiser.training_time_condition(fresh_fb.t), cond=fresh_cond)
        loss_dict = denoiser.loss_fn(process, fresh_fb, fresh_out)
        fresh_loss = loss_dict["total"]
        fresh_scale = float(n_fresh) / float(bsz) if clean_loop_cfg.mode == "drop" else 1.0
        total_loss = fresh_scale * fresh_loss
        aux_loss = aux_contrib = aux_scale = aux_gradient_ratio_actual = None
        fresh_aux_grad_norm = raw_aux_grad_norm = None
        if aux is not None:
            aux_null = (
                torch.full_like(aux.cond, int(model_ctx.model_cfg.num_classes))
                if aux.cond is not None
                else None
            )
            aux_out = denoiser.model_output(
                aux.x,
                aux.t,
                cond=aux.cond,
                null_cond=aux_null,
                guidance_scale=clean_loop_cfg.guidance_scale,
            )
            if clean_loop_cfg.aux_target == "x0":
                aux_prediction = process.x0_from_output(aux.x, aux.t, aux_out, aux={})
                # Store the EMA model's direct x0 output. Reconstructing it as
                # x_t - t*v is algebraically equivalent for linear flow, but
                # unnecessarily routes the target through BF16 velocity
                # conversion and no longer means "native x0" literally.
                aux_target = aux.target_x0
            else:
                aux_prediction = process.velocity_from_output(aux.x, aux.t, aux_out, aux={})
                aux_target = aux.target_v
            aux_loss = regress("mse", aux_prediction, aux_target).mean()
            if clean_loop_cfg.aux_gradient_ratio > 0.0:
                if clean_loop_cfg.aux_gradient_space == "parameter":
                    fresh_grad_norm = _parameter_gradient_norm(fresh_scale * fresh_loss, denoiser)
                    aux_grad_norm = _parameter_gradient_norm(aux_loss, denoiser)
                else:
                    fresh_grad = torch.autograd.grad(fresh_loss, fresh_out, retain_graph=True)[0]
                    aux_grad = torch.autograd.grad(aux_loss, aux_out, retain_graph=True)[0]
                    fresh_grad_norm = fresh_grad.float().norm()
                    aux_grad_norm = aux_grad.float().norm()
                fresh_aux_grad_norm = fresh_grad_norm.detach()
                raw_aux_grad_norm = aux_grad_norm.detach()
                aux_scale = (
                    clean_loop_cfg.aux_gradient_ratio
                    * fresh_grad_norm
                    / aux_grad_norm.clamp_min(1.0e-12)
                ).clamp(max=clean_loop_cfg.aux_scale_max).detach()
                aux_gradient_ratio_actual = (
                    aux_scale * aux_grad_norm / fresh_grad_norm.clamp_min(1.0e-12)
                ).detach()
                aux_contrib = aux_scale * aux_loss
                total_loss = total_loss + aux_contrib
        with torch.no_grad():
            unweighted = _diagnostic_losses(process, fresh_fb, fresh_out)

    extra_metrics: dict[str, torch.Tensor | float] = {
        "fresh/loss": fresh_loss.detach(),
        "fresh/loss_contrib": (fresh_scale * fresh_loss).detach(),
        "clean/fresh_n": float(n_fresh),
        "clean/fresh_scale": fresh_scale,
        "clean/aux_n": float(0 if aux is None else aux.x.shape[0]),
        "clean/bank_size": float(len(clean_loop_bank)),
        "clean/bank_refresh_n": float(bank_refresh_n),
        "clean/aux_gradient_target": clean_loop_cfg.aux_gradient_ratio,
        "clean/aux_gradient_space_parameter": float(clean_loop_cfg.aux_gradient_space == "parameter"),
        "clean/aux_target_x0": float(clean_loop_cfg.aux_target == "x0"),
    }
    if aux is not None and aux_loss is not None and aux_contrib is not None and aux_scale is not None:
        extra_metrics.update({
            "clean/loss_aux": aux_loss.detach(),
            "clean/loss_aux_contrib": aux_contrib.detach(),
            "clean/aux_scale": aux_scale,
            "clean/aux_gradient_ratio_actual": (
                aux_gradient_ratio_actual if aux_gradient_ratio_actual is not None else 0.0
            ),
            "clean/fresh_aux_grad_norm": fresh_aux_grad_norm if fresh_aux_grad_norm is not None else 0.0,
            "clean/raw_aux_grad_norm": raw_aux_grad_norm if raw_aux_grad_norm is not None else 0.0,
            "clean/aux_output_grad_ratio": (
                aux_gradient_ratio_actual
                if clean_loop_cfg.aux_gradient_space == "output" and aux_gradient_ratio_actual is not None
                else 0.0
            ),
            "clean/aux_parameter_grad_ratio": (
                aux_gradient_ratio_actual
                if clean_loop_cfg.aux_gradient_space == "parameter" and aux_gradient_ratio_actual is not None
                else 0.0
            ),
            "clean/solver_index": aux.solver_index.float().mean(),
            "clean/depth": aux.depth.float().mean(),
            "clean/bank_age": (float(step) - aux.producer_step.float().mean()).clamp_min(0.0),
        })
    return TrainForwardBatch(
        loss=total_loss,
        loss_by_target=unweighted,
        batch_size=bsz,
        cond=fresh_cond,
        fb=fresh_fb,
        out=fresh_out,
        extra_metrics=extra_metrics,
    )


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
    grad_norm: torch.Tensor | float | None = None,
) -> None:
    meters.update(loss=fwd.loss.detach(), lr=float(lr))
    for target, val in fwd.loss_by_target.items():
        meters.update(**{f"loss_{target}": val.detach()})
    for key, val in (fwd.extra_metrics or {}).items():
        if isinstance(val, torch.Tensor):
            val = val.detach().float().mean()
        meters.update(**{key: val})
    if grad_norm is not None:
        grad_norm_value = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
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
    if adv_cfg.update_every <= 0 or adv_cfg.d_steps <= 0:
        raise ValueError("adversarial.update_every and adversarial.d_steps must be > 0")

    fb = fwd.fb
    disc_dtype = _discriminator_input_dtype(discriminator)
    if adv_cfg.fake_space == "terminal_x0":
        if any(
            value is None
            for value in (fwd.adv_real, fwd.adv_fake, fwd.adv_t, fwd.adv_output)
        ):
            raise RuntimeError("terminal adversarial payload was not attached to the forward batch")
        real = fwd.adv_real
        fake_g = fwd.adv_fake
        disc_t = fwd.adv_t
        disc_cond = fwd.adv_cond
        adv_output = fwd.adv_output
        assert real is not None and fake_g is not None and disc_t is not None and adv_output is not None
        weights = torch.ones_like(disc_t)
    else:
        n = max(1, int(round(fb.x0.shape[0] * adv_cfg.batch_ratio)))
        real = fb.x0[:n]
        disc_t = fb.t[:n]
        disc_cond = fwd.cond[:n] if fwd.cond is not None else None
        adv_output = fwd.out[:n]
        fake_g = process.x0_from_output(
            fb.xt[:n],
            fb.t[:n],
            adv_output,
            aux={},
        )
        weights = t_weight(disc_t, cfg)
    enabled_fraction = (weights > 0).float().mean()
    metrics: dict[str, torch.Tensor | float] = {
        "gan/g_weight_schedule": g_weight,
        "gan/enabled_t_fraction": enabled_fraction,
        "gan/batch": float(real.shape[0]),
        "gan/fake_space_terminal": float(adv_cfg.fake_space == "terminal_x0"),
    }
    # Per-t GAN diagnostics are meaningful only when every fresh training item
    # has a matching denoising fake. Terminal distribution samples live at t=0
    # and must not be mixed into the fresh time-bin accumulator.
    keep_tbin = adv_cfg.fake_space == "x0_hat" and real.shape[0] == fb.x0.shape[0]
    tbin_values: dict[str, torch.Tensor] = {"advw": weights.detach()} if keep_tbin else {}

    if step % adv_cfg.update_every == 0:
        _set_requires_grad(discriminator, True)
        for _ in range(adv_cfg.d_steps):
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_d = fake_g.detach()
                if adv_cfg.clamp_fake_for_d:
                    fake_d = fake_d.clamp(-1.0, 1.0)
                fake_d = fake_d.to(dtype=disc_dtype)
            use_r1 = adv_cfg.r1_gamma > 0.0 and adv_cfg.r1_interval > 0 and (step % adv_cfg.r1_interval == 0)
            real_images = real.detach().to(dtype=disc_dtype).requires_grad_(use_r1)
            real_logits = discriminator(real_images, disc_t, disc_cond)
            fake_logits = discriminator(fake_d, disc_t, disc_cond)
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
            if keep_tbin:
                tbin_values.update({
                    "drl": per_real.detach(),
                    "dfl": per_fake.detach(),
                    "dacc": 0.5 * ((real_logits.detach() > 0).float() + (fake_logits.detach() < 0).float()),
                })

    _set_requires_grad(discriminator, False)
    fake_logits_g = discriminator(fake_g.to(dtype=disc_dtype), disc_t, disc_cond)
    per_g_adv = generator_loss(fake_logits_g, loss=adv_cfg.loss)
    g_adv = _weighted_mean(per_g_adv, weights)
    if adv_cfg.gradient_ratio > 0.0:
        fresh_grad = torch.autograd.grad(fwd.loss, fwd.out, retain_graph=True)[0]
        adv_grad = torch.autograd.grad(g_adv, adv_output, retain_graph=True)[0]
        fresh_grad_norm = fresh_grad.float().norm()
        adv_grad_norm = adv_grad.float().norm()
        warmup_fraction = min(g_weight / max(adv_cfg.weight, 1.0e-12), 1.0)
        target_ratio = adv_cfg.gradient_ratio * warmup_fraction
        g_scale = (
            target_ratio * fresh_grad_norm / adv_grad_norm.clamp_min(1.0e-12)
        ).detach().clamp(max=adv_cfg.scale_max)
        actual_ratio = g_scale * adv_grad_norm.detach() / fresh_grad_norm.detach().clamp_min(1.0e-12)
        metrics.update({
            "gan/g_scale": g_scale,
            "gan/g_output_grad_ratio": actual_ratio,
            "gan/g_output_grad_target": target_ratio,
            "gan/fresh_output_grad_norm": fresh_grad_norm.detach(),
            "gan/adv_output_grad_norm": adv_grad_norm.detach(),
        })
    else:
        g_scale = float(g_weight)
        metrics["gan/g_scale"] = g_scale
    fwd.loss = fwd.loss + g_scale * g_adv
    metrics["gan/g_adv_loss"] = g_adv.detach()
    if keep_tbin:
        tbin_values["gadv"] = per_g_adv.detach()
    fwd.extra_metrics = {**(fwd.extra_metrics or {}), **metrics}
    if tbin_values:
        fwd.extra_tbin = {**(fwd.extra_tbin or {}), **tbin_values}
    _set_requires_grad(discriminator, True)


def should_log_training_step(*, loop_cfg: LoopConfig, resume: ResumeState) -> bool:
    force_log = resume.run_step <= 20
    return force_log or (resume.global_step % loop_cfg.log_every == 0)


def _configure_parameter_gradient_vjp(cfg: dict) -> bool:
    """Keep compiled backward graphs reusable for parameter-gradient control.

    Functorch's donated-buffer optimization assumes a compiled backward graph is
    consumed once.  Parameter-space gradient matching evaluates two VJPs with
    ``retain_graph=True`` before the real backward, so that optimization must be
    disabled before the model's first compiled forward/backward.
    """

    clean_cfg = cfg.get("clean_loop", {}) or {}
    compile_cfg = cfg.get("compile", {}) or {}
    needs_reusable_backward = (
        bool(compile_cfg.get("enabled", False))
        and bool(clean_cfg.get("enabled", False))
        and int(clean_cfg.get("version", 1)) == 2
        and str(clean_cfg.get("aux_gradient_space", "output")).lower() == "parameter"
    )
    if not needs_reusable_backward:
        return False

    from torch._functorch import config as functorch_config

    functorch_config.donated_buffer = False
    return True


def log_training_step(*, runtime: RuntimeContext, loop_cfg: LoopConfig, resume: ResumeState, meters: MetricLogger, tbin_stats: TimeBinAccumulator, epoch: int, micro_step: int, current_accum_steps: int, progress: ProgressEstimator | None = None) -> None:
    if not should_log_training_step(loop_cfg=loop_cfg, resume=resume):
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
    reusable_compiled_backward = _configure_parameter_gradient_vjp(cfg)
    if reusable_compiled_backward and runtime.is_main:
        runtime.logger.log_text(
            "[compile] functorch.donated_buffer=false for parameter-gradient VJP"
        )
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
    clean_loop_bank = None
    if clean_loop_cfg.enabled:
        clean_loop_bank = (
            TrajectoryBank(clean_loop_cfg, device=runtime.device)
            if clean_loop_cfg.version == 2
            else CleanLoopBank(clean_loop_cfg)
        )
    # Data augmentation pipeline and where it is applied (data_only here).
    augment, augment_mode = build_augment(cfg)
    if runtime.is_main:
        atom_descs = ", ".join(repr(atom) for atom in loss_fn.atoms)
        runtime.logger.log_text(f"[process] name={cfg.get('process', {}).get('name')} output_target={process.output_target} schedule={schedule.mode}")
        runtime.logger.log_text(f"[loss] {atom_descs}")
        loss_weight_shape = _format_loss_weight_shape(loss_fn)
        if loss_weight_shape:
            runtime.logger.log_text("\n".join(loss_weight_shape))
        runtime.logger.log_text(f"[time_sampler] {cfg.get('time_sampler', {'name': 'legacy'})}")
        time_sampler_shape = _format_time_sampler_shape(time_sampler)
        if time_sampler_shape:
            runtime.logger.log_text("\n".join(time_sampler_shape))
        runtime.logger.log_text(f"[time_condition_jitter] {cfg.get('time_condition_jitter', {'enabled': False})}")
        runtime.logger.log_text(f"[model_conditioning] {cfg.get('model_conditioning', {'ignore_time': False})}")
        runtime.logger.log_text(f"[clean_loop] {cfg.get('clean_loop', {'enabled': False})}")
        runtime.logger.log_text(f"[adversarial] {cfg.get('adversarial', {'enabled': False})}")
    fused_optimizer = bool(cfg["train"].get("fused_optimizer", False)) and runtime.device.type == "cuda"
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
        fused=fused_optimizer,
    )
    d_optimizer = None
    if discriminator is not None:
        dc = cfg.get("discriminator", {}) or {}
        betas = dc.get("betas", [0.0, 0.99])
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(dc.get("lr", 2e-4)),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(dc.get("weight_decay", 0.0)),
            fused=fused_optimizer,
        )
    if runtime.is_main:
        runtime.logger.log_text(f"[optimizer] adamw_fused={fused_optimizer}")
    scaler = maybe_make_scaler(precision=model_ctx.precision, use_fsdp=model_ctx.use_fsdp)
    ema = EMA(model=denoiser, decay=float(cfg["train"].get("ema_decay", 0.9999))) if bool(cfg["train"].get("use_ema", True)) else None
    resume = load_resume_state(cfg, denoiser=denoiser, optimizer=optimizer, scaler=scaler, ema=ema, discriminator=discriminator, d_optimizer=d_optimizer, runtime=runtime)
    meters = MetricLogger(window_size=int(cfg["logging"].get("window_size", 20)))
    interval_timer = TrainingIntervalTimer(runtime.device)
    loop_cfg = build_loop_config(cfg, data_ctx.loader, runtime.distributed_cfg)
    if runtime.is_main:
        log_loop_config(runtime.logger, loop_cfg)
    run_step_limit = cfg["train"].get("run_steps")
    run_step_limit = int(run_step_limit) if run_step_limit is not None else None
    display_loop_cfg = loop_cfg
    if run_step_limit is not None:
        display_loop_cfg = replace(loop_cfg, total_steps=resume.global_step + run_step_limit)
    progress = ProgressEstimator(cfg, display_loop_cfg, start_step=resume.global_step)
    tbin_stats = TimeBinAccumulator(num_bins=loop_cfg.tbin_count, device=runtime.device)
    epoch = resume.start_epoch
    end_epoch = loop_cfg.epochs
    if run_step_limit is not None:
        continuation_epochs = math.ceil(run_step_limit / max(1, loop_cfg.optimizer_steps_per_epoch)) + 1
        end_epoch = max(end_epoch, resume.start_epoch + continuation_epochs)

    def _training_limit_reached() -> bool:
        if run_step_limit is not None:
            return resume.run_step >= run_step_limit
        if cfg["train"].get("max_steps") is not None:
            return resume.global_step >= loop_cfg.total_steps
        return False

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
        interval_timer.begin()
        for epoch in range(resume.start_epoch, end_epoch):
            if _training_limit_reached():
                break
            if data_ctx.sampler is not None:
                data_ctx.sampler.set_epoch(epoch)
            use_label_cond = int(cfg["model"].get("num_classes", 0)) > 0
            for micro_step, (x0, y) in enumerate(data_ctx.loader):
                if _training_limit_reached():
                    break
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
                        ema=ema,
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
                interval_timer.record_step(batch_size=fwd.batch_size, world_size=runtime.world_size)
                update_train_meters(meters, fwd, lr=float(optimizer.param_groups[0]["lr"]), grad_norm=grad_norm if did_optimizer_step else None)
                if not did_optimizer_step:
                    continue
                resume.global_step += 1
                resume.run_step += 1
                if should_log_training_step(loop_cfg=display_loop_cfg, resume=resume):
                    # This is the only normal train-loop CUDA synchronization.
                    # It resolves both interval timing and all queued metrics.
                    interval_timer.flush_into(meters)
                log_training_step(runtime=runtime, loop_cfg=display_loop_cfg, resume=resume, meters=meters, tbin_stats=tbin_stats, epoch=epoch, micro_step=micro_step, current_accum_steps=current_accum_steps, progress=progress)
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
                if should_log_training_step(loop_cfg=display_loop_cfg, resume=resume):
                    # Exclude logging/eval/checkpoint work from the next train
                    # interval while still including dataloader idle time.
                    interval_timer.begin()
                if _training_limit_reached():
                    break
            if _training_limit_reached():
                break
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
    except Exception as exc:
        # Preserve the latest optimizer/EMA state for recoverable failures in
        # evaluation, visualization, or logging. Keep the original exception
        # as the one that escapes even if the emergency save also fails.
        if runtime.is_main:
            runtime.logger.log_text(
                f"[checkpoint] {type(exc).__name__}: attempting emergency checkpoint at step {resume.global_step}"
            )
        try:
            _save_final_checkpoint()
        except Exception as save_exc:
            if runtime.is_main:
                runtime.logger.log_text(
                    f"[checkpoint] emergency save failed: {type(save_exc).__name__}: {save_exc}"
                )
        raise
    finally:
        runtime.logger.close()
