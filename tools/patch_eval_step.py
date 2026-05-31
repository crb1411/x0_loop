from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"pattern not found: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    path = Path("x0loop/train.py")
    s = path.read_text()

    # DataContext: add eval_loader.
    if "    eval_loader: DataLoader | None = None\n" not in s:
        s = replace_once(
            s,
            "@dataclass\nclass DataContext:\n    dataset: object\n    sampler: DistributedSampler | None\n    loader: DataLoader\n",
            "@dataclass\nclass DataContext:\n    dataset: object\n    sampler: DistributedSampler | None\n    loader: DataLoader\n    eval_loader: DataLoader | None = None\n",
            label="DataContext.eval_loader",
        )

    # build_dataset: add train flag and use it for torchvision datasets.
    if "def build_dataset(cfg: dict, *, train: bool = True):" not in s:
        s = s.replace("def build_dataset(cfg: dict):", "def build_dataset(cfg: dict, *, train: bool = True):", 1)
        s = s.replace("train=True,\n            download=bool(ds_cfg.get(\"download\", True)),", "train=train,\n            download=bool(ds_cfg.get(\"download\", True)),", 1)
        s = s.replace("train=True,\n            download=bool(ds_cfg.get(\"download\", True)),", "train=train,\n            download=bool(ds_cfg.get(\"download\", True)),", 1)
        s = s.replace("split = ds_cfg.get(\"split\", \"train\")", "split = ds_cfg.get(\"split\", \"train\" if train else \"val\")", 1)

    # build_data_context: build optional eval loader.
    old = """def build_data_context(cfg: dict, runtime: RuntimeContext) -> DataContext:
    dataset = build_dataset(cfg)
    sampler = (
        DistributedSampler(dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True)
        if runtime.is_distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg[\"train\"][\"batch_size\"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg[\"train\"].get(\"num_workers\", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return DataContext(dataset=dataset, sampler=sampler, loader=loader)
"""
    new = """def build_data_context(cfg: dict, runtime: RuntimeContext) -> DataContext:
    dataset = build_dataset(cfg, train=True)
    sampler = (
        DistributedSampler(dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True)
        if runtime.is_distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg[\"train\"][\"batch_size\"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg[\"train\"].get(\"num_workers\", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    eval_loader = None
    eval_cfg = cfg.get(\"eval\", {}) or {}
    if bool(eval_cfg.get(\"enabled\", False)):
        eval_dataset = build_dataset(cfg, train=False)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(eval_cfg.get(\"batch_size\", cfg[\"train\"][\"batch_size\"])),
            shuffle=False,
            sampler=None,
            num_workers=int(eval_cfg.get(\"num_workers\", cfg[\"train\"].get(\"num_workers\", 4))),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        if runtime.is_main:
            runtime.logger.log_text(
                f\"[eval] enabled: batches={len(eval_loader)}, every_steps={int(eval_cfg.get('every_steps', 1000))}, \"
                f\"max_batches={eval_cfg.get('max_batches', 'all')}\"
            )
    return DataContext(dataset=dataset, sampler=sampler, loader=loader, eval_loader=eval_loader)
"""
    if old in s:
        s = s.replace(old, new, 1)

    # Add EvalStats dataclass and helper functions before run_sampling_if_due.
    marker = "\ndef run_sampling_if_due(\n"
    if "class EvalStats:" not in s:
        block = r'''

@dataclass
class EvalStats:
    sums: dict[str, float]
    count: int
    tbin: TimeBinAccumulator


def compute_eval_forward(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    components: TrainComponents,
    x0: torch.Tensor,
    y: object,
    use_label_cond: bool,
) -> dict[str, torch.Tensor | object]:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    t = components.time_sampler.sample(bsz, device=runtime.device)
    cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    fb = components.process.forward_sample(x0=x0, t=t)

    with torch.autocast(
        device_type=runtime.device.type,
        dtype=amp_dtype_for_precision(model_ctx.precision),
        enabled=(model_ctx.precision in {"bf16", "fp16"}),
    ):
        out = model(fb.xt, fb.t, cond=cond)
        loss_dict = components.loss_fn(components.process, fb, out)
        p = components.process
        diag = {
            "loss_weighted": loss_dict["loss_weighted"].detach(),
            "loss_no_weight": loss_dict["loss_no_weight"].detach(),
            "loss_outer_weight": loss_dict["weight"].detach(),
            "loss_eps": regress("mse", p.eps_from_output(fb.xt, fb.t, out, aux=fb.aux), p.eps_target(fb)).detach(),
            "loss_x0": regress("mse", p.x0_from_output(fb.xt, fb.t, out, aux=fb.aux), p.x0_target(fb)).detach(),
            "loss_v": regress("mse", p.v_from_output(fb.xt, fb.t, out, aux=fb.aux), p.v_target(fb)).detach(),
        }
    return {"diag": diag, "fb": fb, "out": out, "batch_size": bsz}


def run_eval_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    components: TrainComponents,
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
                components=components,
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
            tbin.update(schedule=components.schedule, process=components.process, loss_fn=components.loss_fn, fb=fwd["fb"], out=fwd["out"])

    if was_training:
        model.train()

    if total <= 0:
        return
    if runtime.is_main:
        kv = {f"eval/{k}": v / total for k, v in sums.items()}
        kv["eval/num_samples"] = total
        kv["eval/summary"] = tbin.summary(is_distributed=runtime.is_distributed)
        runtime.logger.log_kv(resume.global_step, kv)
'''
        s = s.replace(marker, block + marker, 1)

    # Insert eval call after training log call.
    old_call = """                log_training_step(
                    runtime=runtime,
                    loop_cfg=loop_cfg,
                    resume=resume,
                    meters=meters,
                    tbin_stats=tbin_stats,
                    epoch=epoch,
                    micro_step=micro_step,
                    current_accum_steps=current_accum_steps,
                )
"""
    new_call = old_call + """                run_eval_if_due(
                    cfg=cfg,
                    model=model_ctx.model,
                    runtime=runtime,
                    model_ctx=model_ctx,
                    components=components,
                    data_ctx=data_ctx,
                    resume=resume,
                    use_label_cond=use_label_cond,
                )
"""
    if "run_eval_if_due(\n                    cfg=cfg," not in s and old_call in s:
        s = s.replace(old_call, new_call, 1)

    path.write_text(s)

    # Patch common CIFAR configs.
    yaml_paths = [
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0.yaml"),
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0_weighted.yaml"),
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0_weighted_logit_normal.yaml"),
        Path("train_run/configs/cifar10/cifar10_dit_flow_train_x0_weighted_min.yaml"),
    ]
    eval_block = """
eval:
  enabled: true
  every_steps: 1000
  max_batches: 8
  batch_size: 256
  num_workers: 4
"""
    for yp in yaml_paths:
        if yp.exists():
            y = yp.read_text()
            if "\neval:" not in y:
                insert_before = "\nsample:\n"
                if insert_before in y:
                    y = y.replace(insert_before, eval_block + insert_before, 1)
                else:
                    y = y.rstrip() + "\n" + eval_block
                yp.write_text(y)


if __name__ == "__main__":
    main()
