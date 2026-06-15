from __future__ import annotations

import torch
import torch.distributed as dist

from x0loop.core.process_base import BaseProcess
from x0loop.core.schedules import TimeSchedule
from x0loop.core.time_sampling import TimeSampler
from x0loop.losses.atomic import CompositeLoss, regress
from x0loop.training.context import DataContext, ModelContext, ResumeState, RuntimeContext
from x0loop.training.metrics import TimeBinAccumulator, endpoint_loss_label
from x0loop.training.optimization import amp_dtype_for_precision


def _reduce_eval_totals(
    sums: dict[str, float],
    total: int,
    *,
    device: torch.device,
    is_distributed: bool,
) -> tuple[dict[str, float], int]:
    if not is_distributed or not dist.is_available() or not dist.is_initialized():
        return sums, total

    keys = sorted(sums)
    values = [float(sums[k]) for k in keys]
    values.append(float(total))
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    reduced_sums = {k: float(t[i].item()) for i, k in enumerate(keys)}
    reduced_total = int(t[-1].item())
    return reduced_sums, reduced_total


def compute_eval_forward(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    time_sampler: TimeSampler,
    process: BaseProcess,
    loss_fn: CompositeLoss,
    x0: torch.Tensor,
    y: object,
    use_label_cond: bool,
) -> dict[str, torch.Tensor | object]:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    t = time_sampler.sample(bsz, device=runtime.device)
    cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    fb = process.forward_sample(x0=x0, t=t)

    with torch.autocast(
        device_type=runtime.device.type,
        dtype=amp_dtype_for_precision(model_ctx.precision),
        enabled=(model_ctx.precision in {"bf16", "fp16"}),
    ):
        out = model(fb.xt, fb.t, cond=cond)
        loss_dict = loss_fn(process, fb, out)
        p = process
        terminal_label = endpoint_loss_label(p)
        diag = {
            "loss_weighted": loss_dict["loss_weighted"].detach(),
            "loss_no_weight": loss_dict["loss_no_weight"].detach(),
            "loss_outer_weight": loss_dict["weight"].detach(),
            f"loss_{terminal_label}": regress("mse", p.eps_from_output(fb.xt, fb.t, out, aux={}), p.eps_target(fb)).detach(),
            "loss_x0": regress("mse", p.x0_from_output(fb.xt, fb.t, out, aux={}), p.x0_target(fb)).detach(),
            "loss_v": regress("mse", p.v_from_output(fb.xt, fb.t, out, aux={}), p.v_target(fb)).detach(),
        }
        if hasattr(p, "mu_data") and getattr(p, "predict_mudata", False):
            diag["loss_mudata"] = regress("mse", p.mudata_from_output(fb.xt, fb.t, out, aux={}), p.mudata_target(fb)).detach()
    return {"diag": diag, "fb": fb, "out": out, "batch_size": bsz}


def run_eval_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    schedule: TimeSchedule,
    time_sampler: TimeSampler,
    process: BaseProcess,
    loss_fn: CompositeLoss,
    data_ctx: DataContext,
    resume: ResumeState,
    use_label_cond: bool,
) -> None:
    eval_cfg = cfg.get("eval", {}) or {}
    if not bool(eval_cfg.get("enabled", False)) or data_ctx.eval_loader is None:
        return
    every_steps = int(eval_cfg.get("every_steps", 1000))
    if every_steps <= 0 or resume.global_step <= 0 or (resume.global_step % every_steps != 0):
        return

    max_batches_cfg = eval_cfg.get("max_batches", None)
    max_batches = None if max_batches_cfg in {None, "all"} else int(max_batches_cfg)

    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    total = 0
    tbin = TimeBinAccumulator(num_bins=int(cfg["logging"].get("t_bins", 20)), device=runtime.device)

    with torch.no_grad():
        for bi, batch in enumerate(data_ctx.eval_loader):
            if max_batches is not None and bi >= max_batches:
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x0, y = batch[0], batch[1]
            else:
                x0, y = batch, None
            fwd = compute_eval_forward(
                cfg=cfg,
                model=model,
                runtime=runtime,
                model_ctx=model_ctx,
                time_sampler=time_sampler,
                process=process,
                loss_fn=loss_fn,
                x0=x0,
                y=y,
                use_label_cond=use_label_cond,
            )
            bsz = int(fwd["batch_size"])
            diag = fwd["diag"]
            for k, v in diag.items():
                vv = v
                if isinstance(vv, torch.Tensor):
                    if vv.ndim > 0:
                        vv = vv.mean()
                    sums[k] = sums.get(k, 0.0) + float(vv.item()) * bsz
            total += bsz
            tbin.update(schedule=schedule, process=process, loss_fn=loss_fn, fb=fwd["fb"], out=fwd["out"])

    if was_training:
        model.train()

    sums, total = _reduce_eval_totals(
        sums,
        total,
        device=runtime.device,
        is_distributed=runtime.is_distributed,
    )
    summary = tbin.summary(is_distributed=runtime.is_distributed)

    if total <= 0:
        return
    if runtime.is_main:
        kv = {f"eval/{k}": v / total for k, v in sums.items()}
        kv["eval/num_samples"] = total
        kv["eval/summary"] = summary
        runtime.logger.log_kv(resume.global_step, kv)
