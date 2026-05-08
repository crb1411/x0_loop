from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from x0loop.aug.geom import GeomAugment
from x0loop.aug.identity import NoAug
from x0loop.aug.strong_augment import strongAugment
from x0loop.core.config import DEFAULT_RUNTIME_CONFIG, dump_resolved_config, load_merged_config, resolve_logging_output_dir
from x0loop.core.schedules import TimeSchedule
from x0loop.losses.composite import CompositeLoss
from x0loop.losses.regression import HuberLoss, L1Loss, MSELoss
from x0loop.losses.weighted import WeightedLoss, make_weight_fn
from x0loop.models.dit import DiT, DiTConfig
from x0loop.processes.diffusion_process import DiffusionProcess
from x0loop.processes.flow_process import FlowProcess
from x0loop.utils import dist as dist_utils
from x0loop.utils.checkpoint import load_checkpoint, save_checkpoint
from x0loop.utils.ema import EMA
from x0loop.utils.fsdp import clip_grad_norm, wrap_fsdp2
from x0loop.utils.logger import Logger, MetricLogger


def _maybe_import_vision():
    import torchvision
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10, MNIST, ImageFolder
    from torchvision.utils import make_grid, save_image

    return torchvision, T, CIFAR10, MNIST, ImageFolder, make_grid, save_image


def build_dataset(cfg: dict):
    _, T, CIFAR10, MNIST, ImageFolder, _, _ = _maybe_import_vision()
    ds_cfg = cfg["dataset"]
    img_size = int(cfg["model"]["image_size"])
    tfm = T.Compose(
        [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    name = ds_cfg["name"].lower()
    root = ds_cfg["root"]
    if name == "cifar10":
        train_set = CIFAR10(
            root=root,
            train=True,
            download=bool(ds_cfg.get("download", True)),
            transform=tfm,
        )
    elif name == "mnist":
        train_set = MNIST(
            root=root,
            train=True,
            download=bool(ds_cfg.get("download", True)),
            transform=tfm,
        )
    elif name in {"imagefolder", "tiny-imagenet", "tiny_imagenet"}:
        split = ds_cfg.get("split", "train")
        path = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        train_set = ImageFolder(root=path, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train_set


def build_schedule(cfg: dict) -> TimeSchedule:
    sc = cfg["schedule"]
    return TimeSchedule(mode=sc["mode"], num_steps=int(sc["num_steps"]), diffusion_lambda=float(sc.get("diffusion_lambda", 12.0)))


def build_process(cfg: dict, schedule: TimeSchedule):
    name = cfg["process"]["name"].lower()
    if name == "diffusion":
        pc = cfg.get("process", {})
        return DiffusionProcess(
            schedule=schedule,
            sampler=str(pc.get("sampler", "ddim")),
            posterior_noise_scale=float(pc.get("posterior_noise_scale", 1.0)),
        )
    if name == "flow":
        return FlowProcess(schedule=schedule)
    raise ValueError(f"Unknown process: {name}")


def build_loss(cfg: dict, schedule: TimeSchedule):
    lc = cfg["loss"]
    name = lc["name"].lower()
    if name == "mse":
        return MSELoss()
    if name == "l1":
        return L1Loss()
    if name == "huber":
        return HuberLoss(delta=float(lc.get("delta", 1.0)))
    if name == "weighted":
        inner_name = lc.get("inner", "mse").lower()
        inner = {"mse": MSELoss(), "l1": L1Loss(), "huber": HuberLoss(delta=float(lc.get("delta", 1.0)))}[inner_name]
        weight_fn = make_weight_fn(
            lc.get("weight", "snr"),
            schedule=schedule,
            balance_factor=float(lc.get("balance_factor", 0.5)),
            balance_time=str(lc.get("balance_time", "auto")),
            balance_integral_steps=int(lc.get("balance_integral_steps", 4096)),
        )
        return WeightedLoss(inner=inner, weight_fn=weight_fn)
    if name == "composite":
        losses = []
        weights = []
        for item in lc.get("items", []):
            sub = {"loss": item}
            losses.append(build_loss({"loss": item}, schedule))
            weights.append(float(item.get("weight", 1.0)))
        return CompositeLoss(losses=losses, weights=weights)
    raise ValueError(f"Unknown loss: {name}")


def build_augment(cfg: dict):
    ac = cfg.get("augment", {"name": "none"})
    name = ac.get("name", "none").lower()
    mode = ac.get("mode", "data_only")
    if name == "none":
        return NoAug(), "none"
    if mode != "data_only":
        raise ValueError("augment.mode only supports data_only.")
    if name == "geom":
        aug = GeomAugment(
            hflip_prob=float(ac.get("hflip_prob", 0.5)),
            max_translation=int(ac.get("max_translation", 2)),
            crop_min_scale=float(ac.get("crop_min_scale", 0.9)),
            enable_crop_resize=bool(ac.get("enable_crop_resize", True)),
            random_crop_position=bool(ac.get("random_crop_position", False)),
        )
        return aug, mode
    if name in {"dit", "dit_original"}:
        # DiT-style weak image augmentation (geometric-only branch in this framework).
        aug = GeomAugment(
            hflip_prob=float(ac.get("hflip_prob", 0.5)),
            max_translation=int(ac.get("max_translation", 0)),
            crop_min_scale=float(ac.get("crop_min_scale", 0.9)),
            enable_crop_resize=bool(ac.get("enable_crop_resize", True)),
            random_crop_position=bool(ac.get("random_crop_position", True)),
        )
        return aug, mode
    if name in {"strongaugment", "strong"}:
        # Strong image-model recipe (without label-mixing ops such as mixup/cutmix).
        aug = strongAugment(
            hflip_prob=float(ac.get("hflip_prob", 0.5)),
            crop_min_scale=float(ac.get("crop_min_scale", 0.75)),
            crop_max_scale=float(ac.get("crop_max_scale", 1.0)),
            crop_min_ratio=float(ac.get("crop_min_ratio", 0.75)),
            crop_max_ratio=float(ac.get("crop_max_ratio", 1.3333)),
            brightness=float(ac.get("brightness", 0.4)),
            contrast=float(ac.get("contrast", 0.4)),
            saturation=float(ac.get("saturation", 0.4)),
            grayscale_prob=float(ac.get("grayscale_prob", 0.1)),
            erasing_prob=float(ac.get("erasing_prob", 0.25)),
            erase_min_scale=float(ac.get("erase_min_scale", 0.02)),
            erase_max_scale=float(ac.get("erase_max_scale", 0.2)),
        )
        return aug, mode
    raise ValueError(f"Unknown augment: {name}")


def build_sample_cond(
    cfg: dict,
    *,
    sample_num: int,
    device: torch.device,
    batch_cond: torch.Tensor | None = None,
) -> torch.Tensor | None:
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    if num_classes <= 0:
        return None

    sample_cfg = cfg.get("sample", {})
    labels_cfg = sample_cfg.get("class_labels", None)
    use_batch_cond = bool(sample_cfg.get("use_batch_cond", False))

    cond: torch.Tensor | None = None
    if labels_cfg is not None:
        cond = torch.as_tensor(labels_cfg, device=device, dtype=torch.long).flatten()
    elif use_batch_cond and isinstance(batch_cond, torch.Tensor):
        cond = batch_cond.detach().to(device=device, dtype=torch.long).flatten()
    else:
        # Deterministic default: cycle classes.
        cond = torch.arange(sample_num, device=device, dtype=torch.long) % num_classes

    if cond.numel() == 0:
        return None
    if cond.numel() < sample_num:
        reps = (sample_num + cond.numel() - 1) // cond.numel()
        cond = cond.repeat(reps)
    return cond[:sample_num]


def bucket_losses(t: torch.Tensor, per_example_loss: torch.Tensor) -> dict[str, float]:
    edges = [0.0, 0.1, 0.3, 0.7, 1.0]
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (t >= lo) & (t <= hi)
        else:
            mask = (t >= lo) & (t < hi)
        key = f"loss_t{lo}_{hi}".replace(".", "p")
        if mask.any():
            out[key] = float(per_example_loss[mask].mean().item())
        else:
            out[key] = 0.0
    return out


def compute_unweighted_per_example_loss(loss_fn, out, target, *, t, aux) -> torch.Tensor:
    if isinstance(loss_fn, WeightedLoss):
        return loss_fn.inner.compute_per_example(out, target, t=t, aux=aux)
    if hasattr(loss_fn, "compute_per_example"):
        try:
            return loss_fn.compute_per_example(out, target, t=t, aux=aux)
        except NotImplementedError:
            pass
    return ((out - target) ** 2).view(out.shape[0], -1).mean(dim=1)


def compute_per_example_weight(loss_fn, *, t, aux) -> torch.Tensor:
    if isinstance(loss_fn, WeightedLoss):
        w = loss_fn.weight_fn(t, aux)
        if w.ndim > 1:
            w = w.view(w.shape[0], -1).mean(dim=1)
        return w
    return torch.ones_like(t, dtype=torch.float32)


def compute_tbin_sums(
    t: torch.Tensor,
    per_example_loss: torch.Tensor,
    per_example_weight: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = t.detach().float().clamp(0.0, 1.0)
    loss = per_example_loss.detach().to(torch.float64)
    weight = per_example_weight.detach().to(torch.float64)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=t.device)
    idx = torch.bucketize(t, edges[1:-1], right=False)

    counts = torch.bincount(idx, minlength=num_bins).to(torch.float64)
    sum_loss = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_weight = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_loss.scatter_add_(0, idx, loss)
    sum_weight.scatter_add_(0, idx, weight)
    return counts, sum_weight, sum_loss


def compute_tbin_value_sum(
    t: torch.Tensor,
    per_example_value: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.detach().float().clamp(0.0, 1.0)
    value = per_example_value.detach().to(torch.float64)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=t.device)
    idx = torch.bucketize(t, edges[1:-1], right=False)

    counts = torch.bincount(idx, minlength=num_bins).to(torch.float64)
    sum_value = torch.zeros(num_bins, device=t.device, dtype=torch.float64)
    sum_value.scatter_add_(0, idx, value)
    return counts, sum_value


def compute_per_example_x0_unweighted_loss(process, fb, out) -> torch.Tensor:
    x0_pred = process.x0_from_output(fb.xt.detach(), fb.t.detach(), out.detach(), aux=fb.aux)
    return ((x0_pred - fb.x0.detach()) ** 2).view(x0_pred.shape[0], -1).mean(dim=1)


def compute_dual_x0_loss(process, fb, out, loss_fn) -> torch.Tensor:
    x0_pred = process.x0_from_output(fb.xt, fb.t, out, aux=fb.aux)
    per_example = ((x0_pred - fb.x0) ** 2).view(x0_pred.shape[0], -1).mean(dim=1)
    if isinstance(loss_fn, WeightedLoss):
        w = loss_fn.weight_fn(fb.t, fb.aux)
        if w.ndim > 1:
            w = w.view(w.shape[0], -1).mean(dim=1)
        per_example = per_example * w.to(dtype=per_example.dtype)
    return per_example.mean()


def format_tbin_summary(
    edges: torch.Tensor,
    counts: torch.Tensor,
    avg_a: torch.Tensor,
    avg_w: torch.Tensor,
    loss_per_bin: torch.Tensor,
    x0_loss_per_bin: torch.Tensor | None = None,
) -> str:
    parts = []
    n = counts.numel()
    for i in range(n):
        left = float(edges[i].item())
        right = float(edges[i + 1].item())
        close = "]" if i == n - 1 else ")"
        cnt = int(counts[i].item())
        av = float(avg_a[i].item())
        wv = float(avg_w[i].item())
        leps = float(loss_per_bin[i].item())
        if x0_loss_per_bin is not None:
            lx0 = float(x0_loss_per_bin[i].item())
            parts.append(
                f"[{left:.2f},{right:.2f}{close}: n={cnt}, a={av:.4g}, w={wv:.4g}, leps={leps:.4g}, lx0={lx0:.4g}"
            )
        else:
            parts.append(f"[{left:.2f},{right:.2f}{close}: n={cnt}, a={av:.4g}, w={wv:.4g}, leps={leps:.4g}")
    return " | ".join(parts)


def maybe_make_scaler(precision: str, use_fsdp: bool):
    # use_fsdp kept for API compatibility; GradScaler is unified in torch.amp.
    del use_fsdp
    if precision != "fp16" or not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        # Backward-compatible path for older torch.amp signatures.
        return torch.amp.GradScaler()


def build_step_lr_schedule(train_cfg: dict, total_steps: int, steps_per_epoch: int):
    lr_cfg = train_cfg.get("lr_scheduler", {}) or {}
    name = str(lr_cfg.get("name", "constant")).lower()
    base_lr = float(train_cfg.get("lr", 1e-4))

    if name in {"none", "constant"}:
        def _const_lr(step: int) -> float:
            del step
            return base_lr
        return _const_lr, {"name": "constant", "base_lr": base_lr}

    if name != "cosine":
        raise ValueError(f"Unsupported train.lr_scheduler.name={name}, use constant|cosine")

    max_lr = float(lr_cfg.get("max_lr", base_lr))
    min_lr = float(lr_cfg.get("min_lr", 0.0))

    # New preferred schema:
    # 1) warmup from min_lr -> max_lr for warmup_steps
    # 2) cosine from max_lr -> min_lr for cosine_steps
    # 3) hold min_lr afterwards
    if "warmup_steps" in lr_cfg or "cosine_steps" in lr_cfg:
        warmup_steps = int(lr_cfg.get("warmup_steps", 0))
        cosine_steps = int(lr_cfg.get("cosine_steps", 0))
        for n, v in {"warmup_steps": warmup_steps, "cosine_steps": cosine_steps}.items():
            if v < 0:
                raise ValueError(f"train.lr_scheduler.{n} must be >= 0, got {v}")

        def _warmup_cosine_lr(step: int) -> float:
            s = int(step)
            if warmup_steps > 0 and s < warmup_steps:
                # Linear warmup from min -> max.
                if warmup_steps == 1:
                    return max_lr
                p = float(s) / float(warmup_steps - 1)
                return min_lr + p * (max_lr - min_lr)

            if cosine_steps > 0 and s < warmup_steps + cosine_steps:
                # Cosine decay from max -> min.
                if cosine_steps == 1:
                    return min_lr
                k = s - warmup_steps
                p = float(k) / float(cosine_steps - 1)
                p = min(max(p, 0.0), 1.0)
                return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))

            return min_lr

        meta = {
            "name": "cosine_warmup_hold",
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "cosine_steps": cosine_steps,
            "hold_min_from_step": warmup_steps + cosine_steps,
        }
        return _warmup_cosine_lr, meta

    # Backward-compatible schema:
    # init_steps (init_lr hold) -> max_steps (max_lr hold) -> cosine -> min_steps (min_lr hold)
    init_lr = float(lr_cfg.get("init_lr", base_lr))
    if any(k in lr_cfg for k in ("init_steps", "max_steps", "min_steps")):
        init_steps = int(lr_cfg.get("init_steps", 0))
        max_steps = int(lr_cfg.get("max_steps", 0))
        min_steps = int(lr_cfg.get("min_steps", 0))
    else:
        init_steps = int(lr_cfg.get("init_epochs", 0)) * int(steps_per_epoch)
        max_steps = int(lr_cfg.get("max_epochs", 0)) * int(steps_per_epoch)
        min_steps = int(lr_cfg.get("min_epochs", 0)) * int(steps_per_epoch)

    for n, v in {"init_steps": init_steps, "max_steps": max_steps, "min_steps": min_steps}.items():
        if v < 0:
            raise ValueError(f"train.lr_scheduler.{n} must be >= 0, got {v}")
    occupied = init_steps + max_steps + min_steps
    if occupied > total_steps:
        raise ValueError(
            "train.lr_scheduler init_steps+max_steps+min_steps must be <= total_train_steps, "
            f"got {occupied} > {total_steps}"
        )

    cosine_start = init_steps + max_steps
    cosine_end = total_steps - min_steps
    cosine_steps = max(0, cosine_end - cosine_start)

    def _cosine_lr_legacy(step: int) -> float:
        s = int(step)
        if s < init_steps:
            return init_lr
        if s < cosine_start:
            return max_lr
        if s >= cosine_end:
            return min_lr
        if cosine_steps <= 0:
            return min_lr
        p = float(s - cosine_start + 1) / float(cosine_steps)
        p = min(max(p, 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))

    meta = {
        "name": "cosine_legacy",
        "init_lr": init_lr,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "init_steps": init_steps,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "cosine_steps": cosine_steps,
    }
    return _cosine_lr_legacy, meta


def maybe_compile_model(model: torch.nn.Module, compile_cfg: dict):
    if not bool(compile_cfg.get("enabled", False)):
        return model
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(
        model,
        mode=str(compile_cfg.get("mode", "default")),
        fullgraph=bool(compile_cfg.get("fullgraph", False)),
        dynamic=bool(compile_cfg.get("dynamic", False)),
    )


def save_sample_grid(sample_tensor: torch.Tensor, out_path: str):
    _, _, _, _, _, make_grid, save_image = _maybe_import_vision()
    x = (sample_tensor.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(x, nrow=int(x.shape[0] ** 0.5))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def _tensor_chw_to_uint8_rgb(x: torch.Tensor):
    x = ((x.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c >= 3:
        x = x[:3]
    else:
        pad = torch.zeros((3 - c, h, w), dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    return x.permute(1, 2, 0).contiguous().numpy()


def save_trace_large_images(trace: list[dict], out_dir: str, prefix: str):
    if not trace:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to save trace large images with timestep text.") from exc

    os.makedirs(out_dir, exist_ok=True)
    steps = len(trace)
    bsz = int(trace[0]["x0_hat"].shape[0])
    _, h, w = trace[0]["x0_hat"][0].shape
    cols = int(math.ceil(math.sqrt(steps)))
    rows = int(math.ceil(steps / cols))
    pad = 4
    text_h = 14
    cell_w = w + 2 * pad
    cell_h = h + text_h + 2 * pad
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    t_values = [float(item["t"].item()) if torch.is_tensor(item["t"]) else float(item["t"]) for item in trace]

    for bi in range(bsz):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for si, item in enumerate(trace):
            row = si // cols
            col = si % cols
            x0_hat = item["x0_hat"][bi]
            arr = _tensor_chw_to_uint8_rgb(x0_hat)
            img = Image.fromarray(arr, mode="RGB")
            x0 = col * cell_w + pad
            y0 = row * cell_h + pad
            canvas.paste(img, (x0, y0))
            draw.text((x0, y0 + h + 1), f"t={t_values[si]:.3f}", fill=(0, 0, 0))

        out_path = os.path.join(out_dir, f"{prefix}_sample_{bi:03d}_x0loop.png")
        canvas.save(out_path)


def train(cfg: dict):
    distributed_cfg = cfg.get("distributed", {})
    compile_cfg = cfg.get("compile", {})

    backend = distributed_cfg.get("backend", "nccl")
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
    dist_info = dist_utils.init_distributed(backend=backend)

    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    is_main = dist_info["is_main"]
    is_distributed = dist_info["is_distributed"]

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
        resolved_config_path = dump_resolved_config(cfg, out_dir)
        logger.log_text(f"resolved_config={resolved_config_path}")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dist_utils.seed_everything(int(cfg["train"].get("seed", 42)), rank=rank, deterministic=bool(cfg["train"].get("deterministic", False)))

    dataset = build_dataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model_cfg = DiTConfig(**cfg["model"])
    model = DiT(model_cfg).to(device)
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log_text(f"[model] init: model = DiT(model_cfg).to(device), device={device}")
        logger.log_text(
            "[model] config: "
            f"image_size={model_cfg.image_size}, in_channels={model_cfg.in_channels}, out_channels={model_cfg.out_channels}, "
            f"patch_size={model_cfg.patch_size}, dim={model_cfg.dim}, depth={model_cfg.depth}, heads={model_cfg.heads}, "
            f"mlp_ratio={model_cfg.mlp_ratio}, norm_layer={model_cfg.norm_layer}"
        )
        logger.log_text(
            "[model] shapes: "
            f"input=[B,{model_cfg.in_channels},{model_cfg.image_size},{model_cfg.image_size}], "
            f"output=[B,{model_cfg.out_channels},{model_cfg.image_size},{model_cfg.image_size}], "
            f"tokens={model.num_tokens} ({model.h_tokens}x{model.w_tokens}), token_dim={model_cfg.dim}"
        )
        logger.log_text(f"[model] params: total={total_params:,}, trainable={trainable_params:,}")


    use_fsdp = bool(distributed_cfg.get("fsdp", False) and is_distributed)
    compile_enabled = bool(compile_cfg.get("enabled", False))
    allow_compile_with_fsdp = bool(compile_cfg.get("allow_fsdp", False))

    if compile_enabled and (not use_fsdp or allow_compile_with_fsdp):
        model = maybe_compile_model(model, compile_cfg)
    elif compile_enabled and use_fsdp and (not allow_compile_with_fsdp) and is_main:
        logger.log_text("compile.enabled=true but skipped because FSDP is on. Set compile.allow_fsdp=true to force it.")

    fsdp_mode = "none"
    if use_fsdp:
        model, fsdp_mode = wrap_fsdp2(
            model,
            mixed_precision=distributed_cfg.get("precision", "bf16") in {"bf16", "fp16"},
            precision=distributed_cfg.get("precision", "bf16"),
            use_compile=compile_enabled,
            activation_ckpt=bool(distributed_cfg.get("activation_ckpt", False)),
            device_id=local_rank,
        )
    if is_main:
        logger.log_text(
            f"[runtime] distributed={is_distributed}, world_size={world_size}, use_fsdp={use_fsdp}, "
            f"fsdp_mode={fsdp_mode}, compile={compile_enabled}, precision={distributed_cfg.get('precision', 'bf16')}"
        )

    schedule = build_schedule(cfg)
    process = build_process(cfg, schedule)
    loss_fn = build_loss(cfg, schedule)
    augment, augment_mode = build_augment(cfg)
    loss_cfg = cfg.get("loss", {})
    dual_x0_enabled = bool(loss_cfg.get("dual_x0", loss_cfg.get("dual_loss", False)))
    dual_balance_factor = float(loss_cfg.get("dual_balance_factor", 0.5))
    if not (0.0 <= dual_balance_factor <= 1.0):
        raise ValueError(f"loss.dual_balance_factor must be in [0,1], got {dual_balance_factor}")
    summary_include_x0 = bool(cfg.get("logging", {}).get("summary_include_x0", dual_x0_enabled))
    if is_main and dual_x0_enabled:
        logger.log_text(
            f"[loss] dual_x0 enabled: total=(1-{dual_balance_factor:.4g})*L_eps + {dual_balance_factor:.4g}*L_x0",
        )

    lr = float(cfg["train"].get("lr", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=float(cfg["train"].get("weight_decay", 0.05)))

    precision = str(distributed_cfg.get("precision", "bf16"))
    if device.type == "cpu" and precision in {"bf16", "fp16"}:
        if is_main:
            logger.log_text(f"precision={precision} is not stable on CPU here, fallback to fp32.")
        precision = "fp32"
    scaler = maybe_make_scaler(precision=precision, use_fsdp=use_fsdp)

    ema = None
    if bool(cfg["train"].get("use_ema", True)):
        ema = EMA(model=model, decay=float(cfg["train"].get("ema_decay", 0.9999)))

    resume_path = cfg["train"].get("resume")
    start_epoch = 0
    global_step = 0
    run_step = 0
    ckpt_mode = distributed_cfg.get("checkpoint", {}).get("mode", "full")
    if resume_path:
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            map_location="cpu",
            mode=ckpt_mode,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        if is_main:
            ckpt_keys = ",".join(sorted(ckpt.keys()))
            logger.log_text(
                f"[resume] loaded: path={resume_path}, mode={ckpt_mode}, step={global_step}, epoch={start_epoch}, keys=[{ckpt_keys}]",
            )
    elif is_main:
        logger.log_text("[resume] none (start from scratch)")

    meters = MetricLogger(window_size=int(cfg["logging"].get("window_size", 20)))

    epochs = int(cfg["train"]["epochs"])
    gradient_accumulation_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"train.gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}")
    micro_steps_per_epoch = len(loader)
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / gradient_accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    lr_for_step, lr_sched_meta = build_step_lr_schedule(
        cfg["train"], total_steps=total_steps, steps_per_epoch=optimizer_steps_per_epoch
    )
    grad_clip_cfg = cfg.get("train", {}).get("max_clip_grad", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = cfg.get("train", {}).get("max_grad_norm", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = distributed_cfg.get("grad_clip_norm", 0.0)
    grad_clip = float(grad_clip_cfg)
    if is_main:
        logger.log_text(
            f"[train] gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"micro_steps_per_epoch={micro_steps_per_epoch}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}"
        )
        logger.log_text(f"[train] grad_clip={grad_clip}")
        if lr_sched_meta.get("name") == "cosine_warmup_hold":
            logger.log_text(
                "[train] lr_scheduler=cosine(warmup->cosine->hold_min) "
                f"max_lr={lr_sched_meta['max_lr']:.6g} min_lr={lr_sched_meta['min_lr']:.6g} "
                f"warmup_steps={lr_sched_meta['warmup_steps']} cosine_steps={lr_sched_meta['cosine_steps']} "
                f"hold_min_from_step={lr_sched_meta['hold_min_from_step']}"
            )
        elif lr_sched_meta.get("name") in {"cosine_legacy", "cosine"}:
            logger.log_text(
                "[train] lr_scheduler=cosine(legacy) "
                f"init_lr={lr_sched_meta['init_lr']:.6g} max_lr={lr_sched_meta['max_lr']:.6g} "
                f"min_lr={lr_sched_meta['min_lr']:.6g} init_steps={lr_sched_meta['init_steps']} "
                f"max_steps={lr_sched_meta['max_steps']} min_steps={lr_sched_meta['min_steps']} "
                f"cosine_steps={lr_sched_meta['cosine_steps']}"
            )
        else:
            logger.log_text(f"[train] lr_scheduler=constant lr={lr_sched_meta.get('base_lr', 0.0):.6g}")
    log_every = int(cfg["logging"].get("log_every", 50))
    sample_every = int(cfg["logging"].get("sample_every", 2000))
    save_every = int(distributed_cfg.get("checkpoint", {}).get("every_steps", 2000))
    sample_rank0_only = bool(cfg["logging"].get("sample_rank0_only", True))
    tbin_count = int(cfg["logging"].get("t_bins", 20))
    tbin_edges = torch.linspace(0.0, 1.0, tbin_count + 1, device=device, dtype=torch.float64)
    tbin_counts = torch.zeros(tbin_count, device=device, dtype=torch.float64)
    tbin_sum_alpha = torch.zeros(tbin_count, device=device, dtype=torch.float64)
    tbin_sum_weight = torch.zeros(tbin_count, device=device, dtype=torch.float64)
    tbin_sum_loss = torch.zeros(tbin_count, device=device, dtype=torch.float64)
    tbin_sum_x0_loss = torch.zeros(tbin_count, device=device, dtype=torch.float64)

    model.train()
    iter_start = time.time()

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        use_label_cond = int(cfg["model"].get("num_classes", 0)) > 0

        for micro_step, (x0, y) in enumerate(loader):
            accum_index = micro_step % gradient_accumulation_steps
            update_step = accum_index == 0
            remaining_micro_steps = micro_steps_per_epoch - micro_step
            current_accum_steps = min(gradient_accumulation_steps, remaining_micro_steps)

            if update_step:
                step_lr = float(lr_for_step(global_step))
                for pg in optimizer.param_groups:
                    pg["lr"] = step_lr
                optimizer.zero_grad(set_to_none=True)

            x0 = x0.to(device, non_blocking=True)
            bsz = x0.shape[0]
            t = schedule.sample_t(bsz, device=device)
            cond = y.to(device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None

            if augment_mode == "data_only":
                x0 = augment.apply(x0, augment.sample_params(bsz, device=device))
            fb = process.forward_sample(x0=x0, t=t)

            amp_dtype = torch.float32
            if precision == "bf16":
                amp_dtype = torch.bfloat16
            elif precision == "fp16":
                amp_dtype = torch.float16

            grad_norm = None
            x0_loss = None
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(precision in {"bf16", "fp16"})):
                out = model(fb.xt, fb.t, cond=cond)

                eps_loss = loss_fn(out, fb.target, t=fb.t, aux=fb.aux)
                loss = eps_loss

                if dual_x0_enabled and dual_balance_factor > 0.0:
                    x0_loss = compute_dual_x0_loss(process, fb, out, loss_fn)
                    loss = (1.0 - dual_balance_factor) * eps_loss + dual_balance_factor * x0_loss

                loss_for_backward = loss / float(current_accum_steps)

            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            did_optimizer_step = ((micro_step + 1) % gradient_accumulation_steps == 0) or (
                micro_step + 1 == micro_steps_per_epoch
            )
            if did_optimizer_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        grad_norm = clip_grad_norm(model, grad_clip)
                    else:
                        grad_norm = clip_grad_norm(model, float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        grad_norm = clip_grad_norm(model, grad_clip)
                    else:
                        grad_norm = clip_grad_norm(model, float("inf"))
                    optimizer.step()

            if did_optimizer_step and ema is not None:
                ema.update(model)

            per_example_unweighted = compute_unweighted_per_example_loss(
                loss_fn, out.detach(), fb.target.detach(), t=fb.t.detach(), aux=fb.aux
            )
            per_example_weight = compute_per_example_weight(loss_fn, t=fb.t.detach(), aux=fb.aux)
            c, sw, sl = compute_tbin_sums(
                fb.t.detach(), per_example_unweighted, per_example_weight, num_bins=tbin_count
            )
            alpha_t = fb.aux.get("alpha")
            if alpha_t is None:
                alpha_t = schedule.alpha(fb.t.detach())
            alpha_t = alpha_t.detach()
            if alpha_t.ndim > 1:
                alpha_t = alpha_t.view(alpha_t.shape[0], -1).mean(dim=1)
            _, sa = compute_tbin_value_sum(fb.t.detach(), alpha_t, num_bins=tbin_count)
            if summary_include_x0:
                per_example_x0_unweighted = compute_per_example_x0_unweighted_loss(process, fb, out)
                _, _, sl_x0 = compute_tbin_sums(
                    fb.t.detach(), per_example_x0_unweighted, per_example_weight, num_bins=tbin_count
                )
                tbin_sum_x0_loss += sl_x0
            tbin_counts += c
            tbin_sum_alpha += sa
            tbin_sum_weight += sw
            tbin_sum_loss += sl

            iter_time = time.time() - iter_start
            iter_start = time.time()
            throughput = bsz * world_size / max(iter_time, 1e-6)

            meters.update(
                loss=float(loss.detach().item()),
                loss_eps=float(eps_loss.detach().item()),
                loss_x0=float(x0_loss.detach().item()) if x0_loss is not None else 0.0,
                lr=float(optimizer.param_groups[0]["lr"]),
                iter_s=float(iter_time),
                img_s=float(throughput),
            )
            if did_optimizer_step:
                grad_norm_value = float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                meters.update(grad_norm=grad_norm_value)

            if not did_optimizer_step:
                continue

            global_step += 1
            run_step += 1

            force_log = run_step <= 20
            if force_log or (global_step % log_every == 0):
                meters.reduce_distributed()
                kv = meters.get_log_dict()
                kv["epoch"] = epoch
                kv["micro_step"] = micro_step + 1
                kv["accumulation_steps"] = current_accum_steps
                if grad_clip > 0:
                    kv["grad_clip"] = grad_clip
                if torch.cuda.is_available():
                    kv["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device=device) / (1024**3)

                # Reduce and print unified t-bin stats: count / mean weight / mean unweighted loss.
                rc = tbin_counts.clone()
                rsa = tbin_sum_alpha.clone()
                rsw = tbin_sum_weight.clone()
                rsl = tbin_sum_loss.clone()
                rsxl = tbin_sum_x0_loss.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(rc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rsa, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rsw, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rsl, op=dist.ReduceOp.SUM)
                    if summary_include_x0:
                        dist.all_reduce(rsxl, op=dist.ReduceOp.SUM)
                denom = rc.clamp_min(1.0)
                avg_a = rsa / denom
                avg_w = rsw / denom
                avg_loss = rsl / denom
                avg_x0_loss = None
                if summary_include_x0:
                    avg_x0_loss = rsxl / denom
                summary = format_tbin_summary(tbin_edges, rc, avg_a, avg_w, avg_loss, avg_x0_loss)
                kv["summary"] = summary
                logger.log_kv(global_step, kv, total_steps=total_steps)
                # reset interval accumulators
                tbin_counts.zero_()
                tbin_sum_alpha.zero_()
                tbin_sum_weight.zero_()
                tbin_sum_loss.zero_()
                tbin_sum_x0_loss.zero_()

            do_sample = sample_every > 0 and (global_step % sample_every == 0)
            if do_sample:
                should_run_sample = (not sample_rank0_only) or is_main
                # FSDP forward is collective; run on all ranks when sharded.
                if use_fsdp and is_distributed:
                    should_run_sample = True
                if should_run_sample:
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        if ema is not None:
                            ema.store(model)
                            ema.copy_to(model)
                        sample_num = int(cfg["sample"].get("num", 16))
                        sample_cond = build_sample_cond(
                            cfg,
                            sample_num=sample_num,
                            device=device,
                            batch_cond=cond if use_label_cond else None,
                        )
                        sample_sampler = cfg["sample"].get("sampler", None)
                        if isinstance(sample_sampler, str) and sample_sampler.lower() == "auto":
                            sample_sampler = None
                        result = process.sample(
                            model=model,
                            steps=int(cfg["sample"].get("steps", 50)),
                            shape=(sample_num, model_cfg.out_channels, model_cfg.image_size, model_cfg.image_size),
                            device=device,
                            dtype=torch.float32,
                            return_trace=True,
                            cond=sample_cond,
                            sampler=sample_sampler,
                            posterior_noise_scale=cfg["sample"].get("posterior_noise_scale", None),
                        )
                        if ema is not None:
                            ema.restore(model)

                    if is_main:
                        sample_dir = os.path.join(out_dir, "samples")
                        save_trace_large_images(result.get("trace", []), sample_dir, f"step_{global_step:08d}")
                        if bool(cfg["sample"].get("save_trace", False)) and "trace" in result:
                            trace_path = os.path.join(sample_dir, f"step_{global_step:08d}_trace.pt")
                            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                            torch.save(result["trace"], trace_path)

                    if was_training:
                        model.train()
                if is_distributed:
                    dist_utils.barrier()

            if save_every > 0 and (global_step % save_every == 0):
                ckpt_path = os.path.join(out_dir, "checkpoints", f"ckpt_step_{global_step:08d}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    step=global_step,
                    epoch=epoch,
                    config=cfg,
                    is_main=is_main,
                    mode=ckpt_mode,
                )
                if is_distributed:
                    dist_utils.barrier()

    logger.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="x0loop/configs/default.yaml")
    parser.add_argument("--runtime-config", type=str, default=DEFAULT_RUNTIME_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_merged_config(args.config, args.runtime_config, resolve_logging=False)
    train(cfg)
