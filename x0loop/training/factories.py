from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from x0loop.aug.geom import GeomAugment
from x0loop.aug.identity import NoAug
from x0loop.aug.base import BaseAugment
from x0loop.aug.strong_augment import strongAugment
from x0loop.core.config import dump_resolved_config, resolve_logging_output_dir
from x0loop.core.process_base import BaseProcess
from x0loop.core.schedules import TimeSchedule
from x0loop.models.factory import build_model
from x0loop.models.denoiser import Denoiser
from x0loop.processes.diffusion_process import DiffusionProcess
from x0loop.processes.flow_process import FlowProcess
from x0loop.training.context import DataContext, ModelContext, ResumeState, RuntimeContext
from x0loop.training.optimization import maybe_compile_model
from x0loop.utils import dist as dist_utils
from x0loop.utils.checkpoint import load_checkpoint
from x0loop.utils.ema import EMA
from x0loop.utils.fsdp import wrap_fsdp2
from x0loop.utils.logger import Logger


def _maybe_import_vision():
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10, MNIST, ImageFolder
    return T, CIFAR10, MNIST, ImageFolder


def build_dataset(cfg: dict, *, train: bool = True):
    T, CIFAR10, MNIST, ImageFolder = _maybe_import_vision()
    ds_cfg = cfg["dataset"]
    img_size = int(cfg["model"]["image_size"])
    tfm = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor(), T.Lambda(lambda x: x * 2.0 - 1.0)])
    name = ds_cfg["name"].lower()
    root = ds_cfg["root"]
    if name == "cifar10":
        return CIFAR10(root=root, train=train, download=bool(ds_cfg.get("download", True)), transform=tfm)
    if name == "mnist":
        return MNIST(root=root, train=train, download=bool(ds_cfg.get("download", True)), transform=tfm)
    if name in {"imagefolder", "tiny-imagenet", "tiny_imagenet"}:
        split = ds_cfg.get("split", "train" if train else "val")
        path = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        return ImageFolder(root=path, transform=tfm)
    raise ValueError(f"Unsupported dataset: {name}")


def build_schedule(cfg: dict) -> TimeSchedule:
    sc = cfg["schedule"]
    return TimeSchedule(mode=sc["mode"], num_steps=int(sc.get("num_steps", 1000)), beta_min=float(sc.get("beta_min", 0.1)), beta_max=float(sc.get("beta_max", 20.0)))


def build_process(cfg: dict, schedule: TimeSchedule) -> BaseProcess:
    pc = cfg.get("process", {})
    name = str(pc.get("name", "diffusion")).lower()
    if name != str(schedule.mode).lower():
        raise ValueError(f"process.name ({name}) must match schedule.mode ({schedule.mode}).")
    output_target = str(pc.get("output_target", "eps")).lower()
    if name == "diffusion":
        return DiffusionProcess(schedule=schedule, output_target=output_target, sampler=str(pc.get("sampler", "ddim")), posterior_noise_scale=float(pc.get("posterior_noise_scale", 1.0)))
    if name == "flow":
        return FlowProcess(schedule=schedule, output_target=output_target, sampler=str(pc.get("sampler", "euler")))
    raise ValueError(f"Unknown process: {name}")


def build_augment(cfg: dict) -> tuple[BaseAugment, str]:
    ac = cfg.get("augment", {"name": "none"})
    name = ac.get("name", "none").lower()
    mode = ac.get("mode", "data_only")
    if name == "none":
        return NoAug(), "none"
    if mode != "data_only":
        raise ValueError("augment.mode only supports data_only.")
    if name == "geom":
        return GeomAugment(hflip_prob=float(ac.get("hflip_prob", 0.5)), max_translation=int(ac.get("max_translation", 2)), crop_min_scale=float(ac.get("crop_min_scale", 0.9)), enable_crop_resize=bool(ac.get("enable_crop_resize", True)), random_crop_position=bool(ac.get("random_crop_position", False))), mode
    if name in {"dit", "dit_original"}:
        return GeomAugment(hflip_prob=float(ac.get("hflip_prob", 0.5)), max_translation=int(ac.get("max_translation", 0)), crop_min_scale=float(ac.get("crop_min_scale", 0.9)), enable_crop_resize=bool(ac.get("enable_crop_resize", True)), random_crop_position=bool(ac.get("random_crop_position", True))), mode
    if name in {"strongaugment", "strong"}:
        return strongAugment(hflip_prob=float(ac.get("hflip_prob", 0.5)), crop_min_scale=float(ac.get("crop_min_scale", 0.75)), crop_max_scale=float(ac.get("crop_max_scale", 1.0)), crop_min_ratio=float(ac.get("crop_min_ratio", 0.75)), crop_max_ratio=float(ac.get("crop_max_ratio", 1.3333)), brightness=float(ac.get("brightness", 0.4)), contrast=float(ac.get("contrast", 0.4)), saturation=float(ac.get("saturation", 0.4)), grayscale_prob=float(ac.get("grayscale_prob", 0.1)), erasing_prob=float(ac.get("erasing_prob", 0.25)), erase_min_scale=float(ac.get("erase_min_scale", 0.02)), erase_max_scale=float(ac.get("erase_max_scale", 0.2))), mode
    raise ValueError(f"Unknown augment: {name}")


def init_runtime(cfg: dict) -> RuntimeContext:
    distributed_cfg = cfg.get("distributed", {})
    compile_cfg = cfg.get("compile", {})
    backend = distributed_cfg.get("backend", "nccl")
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
    dist_info = dist_utils.init_distributed(backend=backend)
    is_main = bool(dist_info["is_main"])
    is_distributed = bool(dist_info["is_distributed"])
    run_timestamp = os.environ.get("X0LOOP_RUN_TIMESTAMP", "")
    if is_main and not run_timestamp:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if is_distributed:
        payload = [run_timestamp]
        dist.broadcast_object_list(payload, src=0)
        run_timestamp = payload[0]
    os.environ["X0LOOP_RUN_TIMESTAMP"] = run_timestamp
    resolve_logging_output_dir(cfg, timestamp=run_timestamp)
    out_dir = cfg["logging"]["out_dir"]
    logger = Logger(out_dir=out_dir, is_main=is_main, use_tb=bool(cfg["logging"].get("use_tb", True)))
    if is_main:
        logger.log_text(f"output_dir={out_dir}")
        logger.log_text(f"resolved_config={dump_resolved_config(cfg, out_dir)}")
    local_rank = int(dist_info["local_rank"])
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dist_utils.seed_everything(int(cfg["train"].get("seed", 42)), rank=int(dist_info["rank"]), deterministic=bool(cfg["train"].get("deterministic", False)))
    return RuntimeContext(distributed_cfg=distributed_cfg, compile_cfg=compile_cfg, dist_info=dist_info, device=device, out_dir=out_dir, logger=logger)


def build_data_context(cfg: dict, runtime: RuntimeContext) -> DataContext:
    dataset = build_dataset(cfg, train=True)
    sampler = DistributedSampler(dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True) if runtime.is_distributed else None
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=(sampler is None), sampler=sampler, num_workers=int(cfg["train"].get("num_workers", 4)), pin_memory=torch.cuda.is_available(), drop_last=True)
    eval_loader = None
    eval_cfg = cfg.get("eval", {}) or {}
    if bool(eval_cfg.get("enabled", False)):
        eval_dataset = build_dataset(cfg, train=False)
        eval_loader = DataLoader(eval_dataset, batch_size=int(eval_cfg.get("batch_size", cfg["train"]["batch_size"])), shuffle=False, num_workers=int(eval_cfg.get("num_workers", cfg["train"].get("num_workers", 4))), pin_memory=torch.cuda.is_available(), drop_last=False)
        if runtime.is_main:
            runtime.logger.log_text(f"[eval] enabled: batches={len(eval_loader)}, every_steps={int(eval_cfg.get('every_steps', 1000))}, max_batches={eval_cfg.get('max_batches', 'all')}")
    return DataContext(dataset=dataset, sampler=sampler, loader=loader, eval_loader=eval_loader)


def log_model_summary(logger: Logger, model: torch.nn.Module, model_cfg: object, device: torch.device) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_text(f"[model] init: model = {type(model).__name__}(model_cfg).to(device), device={device}")
    logger.log_text(f"[model] config: {model_cfg}")
    shape_desc = f"[model] shapes: input=[B,{model_cfg.in_channels},{model_cfg.image_size},{model_cfg.image_size}], output=[B,{model_cfg.out_channels},{model_cfg.image_size},{model_cfg.image_size}]"
    if all(hasattr(model, k) for k in ("num_tokens", "h_tokens", "w_tokens")):
        token_dim = getattr(model_cfg, "dim", getattr(model_cfg, "base_channels", "n/a"))
        shape_desc += f", tokens={model.num_tokens} ({model.h_tokens}x{model.w_tokens}), token_dim={token_dim}"
    logger.log_text(shape_desc)
    logger.log_text(f"[model] params: total={total_params:,}, trainable={trainable_params:,}")


def build_model_context(cfg: dict, runtime: RuntimeContext) -> ModelContext:
    model, model_cfg = build_model(cfg["model"])
    model = model.to(runtime.device)
    if runtime.is_main:
        log_model_summary(runtime.logger, model, model_cfg, runtime.device)
    use_fsdp = bool(runtime.distributed_cfg.get("fsdp", False) and runtime.is_distributed)
    compile_enabled = bool(runtime.compile_cfg.get("enabled", False))
    allow_compile_with_fsdp = bool(runtime.compile_cfg.get("allow_fsdp", False))
    if compile_enabled and (not use_fsdp or allow_compile_with_fsdp):
        model = maybe_compile_model(model, runtime.compile_cfg)
    elif compile_enabled and use_fsdp and (not allow_compile_with_fsdp) and runtime.is_main:
        runtime.logger.log_text("compile.enabled=true but skipped because FSDP is on. Set compile.allow_fsdp=true to force it.")
    fsdp_mode = "none"
    if use_fsdp:
        model, fsdp_mode = wrap_fsdp2(model, mixed_precision=runtime.distributed_cfg.get("precision", "bf16") in {"bf16", "fp16"}, precision=runtime.distributed_cfg.get("precision", "bf16"), use_compile=compile_enabled, activation_ckpt=bool(runtime.distributed_cfg.get("activation_ckpt", False)), device_id=runtime.local_rank)
    precision = str(runtime.distributed_cfg.get("precision", "bf16"))
    if runtime.device.type == "cpu" and precision in {"bf16", "fp16"}:
        if runtime.is_main:
            runtime.logger.log_text(f"precision={precision} is not stable on CPU here, fallback to fp32.")
        precision = "fp32"
    if runtime.is_main:
        runtime.logger.log_text(f"[runtime] distributed={runtime.is_distributed}, world_size={runtime.world_size}, use_fsdp={use_fsdp}, fsdp_mode={fsdp_mode}, compile={compile_enabled}, precision={precision}")
    return ModelContext(model=model, model_cfg=model_cfg, use_fsdp=use_fsdp, fsdp_mode=fsdp_mode, precision=precision)


def load_resume_state(
    cfg: dict,
    *,
    denoiser: Denoiser,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    ema: EMA | None,
    runtime: RuntimeContext,
) -> ResumeState:
    resume_path = cfg["train"].get("resume")
    ckpt_mode = runtime.distributed_cfg.get("checkpoint", {}).get("mode", "full")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=denoiser, optimizer=optimizer, scaler=scaler, ema=ema, map_location="cpu", mode=ckpt_mode)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        if runtime.is_main:
            ckpt_keys = ",".join(sorted(ckpt.keys()))
            runtime.logger.log_text(f"[resume] loaded: path={resume_path}, mode={ckpt_mode}, step={global_step}, epoch={start_epoch}, keys=[{ckpt_keys}]")
        return ResumeState(start_epoch=start_epoch, global_step=global_step, run_step=0, ckpt_mode=ckpt_mode)
    if runtime.is_main:
        runtime.logger.log_text("[resume] none (start from scratch)")
    return ResumeState(start_epoch=0, global_step=0, run_step=0, ckpt_mode=ckpt_mode)
