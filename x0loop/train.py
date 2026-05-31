from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from x0loop.core.time_sampling import build_time_sampler
from x0loop.losses.spec import build_loss as _build_loss
from x0loop.losses.atomic import AtomicLoss, CompositeLoss, regress
from x0loop.models.dit import DiT, DiTConfig
from x0loop.processes.diffusion_process import DiffusionProcess
from x0loop.processes.flow_process import FlowProcess
from x0loop.utils import dist as dist_utils
from x0loop.utils.checkpoint import load_checkpoint, save_checkpoint
from x0loop.utils.ema import EMA
from x0loop.utils.fsdp import clip_grad_norm, wrap_fsdp2
from x0loop.utils.logger import Logger, MetricLogger


@dataclass
class RuntimeContext:
    distributed_cfg: dict
    compile_cfg: dict
    dist_info: dict
    device: torch.device
    out_dir: str
    logger: Logger

    @property
    def rank(self) -> int:
        return int(self.dist_info["rank"])

    @property
    def local_rank(self) -> int:
        return int(self.dist_info["local_rank"])

    @property
    def world_size(self) -> int:
        return int(self.dist_info["world_size"])

    @property
    def is_main(self) -> bool:
        return bool(self.dist_info["is_main"])

    @property
    def is_distributed(self) -> bool:
        return bool(self.dist_info["is_distributed"])


@dataclass
class DataContext:
    dataset: object
    sampler: DistributedSampler | None
    loader: DataLoader


@dataclass
class ModelContext:
    model: torch.nn.Module
    model_cfg: DiTConfig
    use_fsdp: bool
    fsdp_mode: str
    precision: str


@dataclass
class TrainComponents:
    schedule: TimeSchedule
    time_sampler: object
    process: object
    loss_fn: CompositeLoss
    augment: object
    augment_mode: str
    optimizer: torch.optim.Optimizer
    scaler: object | None
    ema: EMA | None


@dataclass
class ResumeState:
    start_epoch: int
    global_step: int
    run_step: int
    ckpt_mode: str


@dataclass
class LoopConfig:
    epochs: int
    gradient_accumulation_steps: int
    micro_steps_per_epoch: int
    optimizer_steps_per_epoch: int
    total_steps: int
    lr_for_step: object
    lr_sched_meta: dict
    grad_clip: float
    log_every: int
    sample_every: int
    save_every: int
    sample_rank0_only: bool
    tbin_count: int


@dataclass
class ForwardBatch:
    loss: torch.Tensor
    loss_by_target: dict  # {"eps": Tensor, "x0": Tensor, ...}
    batch_size: int
    cond: torch.Tensor | None
    fb: object
    out: torch.Tensor


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
    return TimeSchedule(mode=sc["mode"], num_steps=int(sc.get("num_steps", 1000)),
                        beta_min=float(sc.get("beta_min", 0.1)), beta_max=float(sc.get("beta_max", 20.0)))


def build_process(cfg: dict, schedule: TimeSchedule):
    pc = cfg.get("process", {})
    name = str(pc.get("name", "diffusion")).lower()
    output_target = str(pc.get("output_target", "eps")).lower()
    if name == "diffusion":
        return DiffusionProcess(
            schedule=schedule,
            output_target=output_target,
            sampler=str(pc.get("sampler", "ddim")),
            posterior_noise_scale=float(pc.get("posterior_noise_scale", 1.0)),
        )
    if name == "flow":
        return FlowProcess(schedule=schedule, output_target=output_target)
    raise ValueError(f"Unknown process: {name}")



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


def build_null_class_cond(cfg: dict, *, sample_num: int, device: torch.device) -> torch.Tensor | None:
    num_classes = int(cfg.get("model", {}).get("num_classes", 0))
    if num_classes <= 0:
        return None
    return torch.full((sample_num,), num_classes, device=device, dtype=torch.long)


def apply_classifier_free_label_dropout(cond: torch.Tensor | None, *, null_class_id: int, drop_prob: float) -> torch.Tensor | None:
    if cond is None or drop_prob <= 0.0:
        return cond
    if not (0.0 <= drop_prob <= 1.0):
        raise ValueError(f"train.class_dropout_prob must be in [0,1], got {drop_prob}")
    drop_mask = torch.rand(cond.shape, device=cond.device) < drop_prob
    return torch.where(drop_mask, torch.full_like(cond, null_class_id), cond)


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



def format_tbin_summary(
    edges: torch.Tensor,
    counts: torch.Tensor,
    avg_a: torch.Tensor,
    avg_w: torch.Tensor,
    avg_eps: torch.Tensor,
    avg_x0: torch.Tensor,
    avg_v: torch.Tensor,
) -> str:
    parts = []
    n = counts.numel()
    for i in range(n):
        left = float(edges[i].item())
        right = float(edges[i + 1].item())
        close = "]" if i == n - 1 else ")"
        cnt = int(counts[i].item())
        fields = [
            f"n={cnt}",
            f"a={float(avg_a[i].item()):.4g}",
            f"w={float(avg_w[i].item()):.4g}",
            f"leps={float(avg_eps[i].item()):.4g}",
            f"lx0={float(avg_x0[i].item()):.4g}",
            f"lv={float(avg_v[i].item()):.4g}",
        ]
        parts.append(f"[{left:.2f},{right:.2f}{close}: {', '.join(fields)}")
    return " | ".join(parts)


class TimeBinAccumulator:
    def __init__(self, *, num_bins: int, device: torch.device):
        self.num_bins = int(num_bins)
        self.edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device=device, dtype=torch.float64)
        self.counts = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_alpha = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_weight = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_eps = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_x0 = torch.zeros(self.num_bins, device=device, dtype=torch.float64)
        self.sum_v = torch.zeros(self.num_bins, device=device, dtype=torch.float64)

    def update(self, *, schedule, process, loss_fn: CompositeLoss, fb, out) -> None:
        t = fb.t.detach()
        out_d = out.detach()

        # Per-example unweighted MSE for eps/x0/v (diagnostic, always MSE regardless of training formula).
        eps_u = regress("mse", process.eps_from_output(fb.xt, t, out_d, aux=fb.aux), process.eps_target(fb).detach())
        x0_u = regress("mse", process.x0_from_output(fb.xt, t, out_d, aux=fb.aux), process.x0_target(fb).detach())
        v_u = regress("mse", process.v_from_output(fb.xt, t, out_d, aux=fb.aux), process.v_target(fb).detach())

        # Weight from first atom that has a weight_fn, else uniform.
        ref_atom = next((a for a in loss_fn.atoms if a.weight_fn is not None), None)
        if ref_atom is not None:
            w = ref_atom.weight_fn(t, fb.aux)
            if w.ndim > 1:
                w = w.view(w.shape[0], -1).mean(dim=1)
            w = w.float()
        else:
            w = torch.ones(t.shape[0], device=t.device, dtype=torch.float32)

        alpha_t = fb.aux.get("alpha")
        if alpha_t is None:
            alpha_t = schedule.alpha(t)
        alpha_t = alpha_t.detach().float()
        if alpha_t.ndim > 1:
            alpha_t = alpha_t.view(alpha_t.shape[0], -1).mean(dim=1)

        c, sw, sl_eps = compute_tbin_sums(t, eps_u, w, num_bins=self.num_bins)
        _, _, sl_x0 = compute_tbin_sums(t, x0_u, w, num_bins=self.num_bins)
        _, _, sl_v = compute_tbin_sums(t, v_u, w, num_bins=self.num_bins)
        _, sa = compute_tbin_value_sum(t, alpha_t, num_bins=self.num_bins)

        self.counts += c
        self.sum_alpha += sa
        self.sum_weight += sw
        self.sum_eps += sl_eps
        self.sum_x0 += sl_x0
        self.sum_v += sl_v

    def summary(self, *, is_distributed: bool) -> str:
        rc = self.counts.clone()
        rsa = self.sum_alpha.clone()
        rsw = self.sum_weight.clone()
        rse = self.sum_eps.clone()
        rsxl = self.sum_x0.clone()
        rsvl = self.sum_v.clone()
        if is_distributed and dist.is_available() and dist.is_initialized():
            for t in (rc, rsa, rsw, rse, rsxl, rsvl):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        denom = rc.clamp_min(1.0)
        return format_tbin_summary(
            self.edges, rc,
            rsa / denom, rsw / denom,
            rse / denom, rsxl / denom, rsvl / denom,
        )

    def reset(self) -> None:
        for t in (self.counts, self.sum_alpha, self.sum_weight, self.sum_eps, self.sum_x0, self.sum_v):
            t.zero_()


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
            img = Image.fromarray(arr)
            x0 = col * cell_w + pad
            y0 = row * cell_h + pad
            canvas.paste(img, (x0, y0))
            draw.text((x0, y0 + h + 1), f"t={t_values[si]:.3f}", fill=(0, 0, 0))

        out_path = os.path.join(out_dir, f"{prefix}_sample_{bi:03d}_x0loop.png")
        canvas.save(out_path)


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
        resolved_config_path = dump_resolved_config(cfg, out_dir)
        logger.log_text(f"resolved_config={resolved_config_path}")

    local_rank = int(dist_info["local_rank"])
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dist_utils.seed_everything(
        int(cfg["train"].get("seed", 42)),
        rank=int(dist_info["rank"]),
        deterministic=bool(cfg["train"].get("deterministic", False)),
    )

    return RuntimeContext(
        distributed_cfg=distributed_cfg,
        compile_cfg=compile_cfg,
        dist_info=dist_info,
        device=device,
        out_dir=out_dir,
        logger=logger,
    )


def build_data_context(cfg: dict, runtime: RuntimeContext) -> DataContext:
    dataset = build_dataset(cfg)
    sampler = (
        DistributedSampler(dataset, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=True)
        if runtime.is_distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return DataContext(dataset=dataset, sampler=sampler, loader=loader)


def log_model_summary(logger: Logger, model: torch.nn.Module, model_cfg: DiTConfig, device: torch.device) -> None:
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


def build_model_context(cfg: dict, runtime: RuntimeContext) -> ModelContext:
    model_cfg = DiTConfig(**cfg["model"])
    model = DiT(model_cfg).to(runtime.device)
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
        model, fsdp_mode = wrap_fsdp2(
            model,
            mixed_precision=runtime.distributed_cfg.get("precision", "bf16") in {"bf16", "fp16"},
            precision=runtime.distributed_cfg.get("precision", "bf16"),
            use_compile=compile_enabled,
            activation_ckpt=bool(runtime.distributed_cfg.get("activation_ckpt", False)),
            device_id=runtime.local_rank,
        )

    precision = str(runtime.distributed_cfg.get("precision", "bf16"))
    if runtime.device.type == "cpu" and precision in {"bf16", "fp16"}:
        if runtime.is_main:
            runtime.logger.log_text(f"precision={precision} is not stable on CPU here, fallback to fp32.")
        precision = "fp32"

    if runtime.is_main:
        runtime.logger.log_text(
            f"[runtime] distributed={runtime.is_distributed}, world_size={runtime.world_size}, use_fsdp={use_fsdp}, "
            f"fsdp_mode={fsdp_mode}, compile={compile_enabled}, precision={precision}"
        )

    return ModelContext(model=model, model_cfg=model_cfg, use_fsdp=use_fsdp, fsdp_mode=fsdp_mode, precision=precision)


def build_train_components(cfg: dict, model_ctx: ModelContext, runtime: RuntimeContext) -> TrainComponents:
    schedule = build_schedule(cfg)
    time_sampler = build_time_sampler(cfg, schedule)
    process = build_process(cfg, schedule)
    loss_fn = _build_loss(cfg["loss"], schedule)
    augment, augment_mode = build_augment(cfg)

    if runtime.is_main:
        atom_descs = ", ".join(repr(a) for a in loss_fn.atoms)
        runtime.logger.log_text(f"[loss] {atom_descs}")
        runtime.logger.log_text(f"[time_sampler] {cfg.get('time_sampler', {'name': 'legacy'})}")

    optimizer = torch.optim.AdamW(
        model_ctx.model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-4)),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )
    scaler = maybe_make_scaler(precision=model_ctx.precision, use_fsdp=model_ctx.use_fsdp)
    ema = EMA(model=model_ctx.model, decay=float(cfg["train"].get("ema_decay", 0.9999))) if bool(cfg["train"].get("use_ema", True)) else None

    return TrainComponents(
        schedule=schedule,
        time_sampler=time_sampler,
        process=process,
        loss_fn=loss_fn,
        augment=augment,
        augment_mode=augment_mode,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
    )


def load_resume_state(cfg: dict, model_ctx: ModelContext, components: TrainComponents, runtime: RuntimeContext) -> ResumeState:
    resume_path = cfg["train"].get("resume")
    ckpt_mode = runtime.distributed_cfg.get("checkpoint", {}).get("mode", "full")
    if resume_path:
        ckpt = load_checkpoint(
            resume_path,
            model=model_ctx.model,
            optimizer=components.optimizer,
            scaler=components.scaler,
            ema=components.ema,
            map_location="cpu",
            mode=ckpt_mode,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        if runtime.is_main:
            ckpt_keys = ",".join(sorted(ckpt.keys()))
            runtime.logger.log_text(
                f"[resume] loaded: path={resume_path}, mode={ckpt_mode}, step={global_step}, epoch={start_epoch}, keys=[{ckpt_keys}]",
            )
        return ResumeState(start_epoch=start_epoch, global_step=global_step, run_step=0, ckpt_mode=ckpt_mode)

    if runtime.is_main:
        runtime.logger.log_text("[resume] none (start from scratch)")
    return ResumeState(start_epoch=0, global_step=0, run_step=0, ckpt_mode=ckpt_mode)


def build_loop_config(cfg: dict, loader: DataLoader, distributed_cfg: dict) -> LoopConfig:
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

    return LoopConfig(
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        micro_steps_per_epoch=micro_steps_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_steps=total_steps,
        lr_for_step=lr_for_step,
        lr_sched_meta=lr_sched_meta,
        grad_clip=float(grad_clip_cfg),
        log_every=int(cfg["logging"].get("log_every", 50)),
        sample_every=int(cfg["logging"].get("sample_every", 2000)),
        save_every=int(distributed_cfg.get("checkpoint", {}).get("every_steps", 2000)),
        sample_rank0_only=bool(cfg["logging"].get("sample_rank0_only", True)),
        tbin_count=int(cfg["logging"].get("t_bins", 20)),
    )


def log_loop_config(logger: Logger, loop_cfg: LoopConfig) -> None:
    logger.log_text(
        f"[train] gradient_accumulation_steps={loop_cfg.gradient_accumulation_steps}, "
        f"micro_steps_per_epoch={loop_cfg.micro_steps_per_epoch}, optimizer_steps_per_epoch={loop_cfg.optimizer_steps_per_epoch}"
    )
    logger.log_text(f"[train] grad_clip={loop_cfg.grad_clip}")
    meta = loop_cfg.lr_sched_meta
    if meta.get("name") == "cosine_warmup_hold":
        logger.log_text(
            "[train] lr_scheduler=cosine(warmup->cosine->hold_min) "
            f"max_lr={meta['max_lr']:.6g} min_lr={meta['min_lr']:.6g} "
            f"warmup_steps={meta['warmup_steps']} cosine_steps={meta['cosine_steps']} "
            f"hold_min_from_step={meta['hold_min_from_step']}"
        )
    elif meta.get("name") in {"cosine_legacy", "cosine"}:
        logger.log_text(
            "[train] lr_scheduler=cosine(legacy) "
            f"init_lr={meta['init_lr']:.6g} max_lr={meta['max_lr']:.6g} "
            f"min_lr={meta['min_lr']:.6g} init_steps={meta['init_steps']} "
            f"max_steps={meta['max_steps']} min_steps={meta['min_steps']} "
            f"cosine_steps={meta['cosine_steps']}"
        )
    else:
        logger.log_text(f"[train] lr_scheduler=constant lr={meta.get('base_lr', 0.0):.6g}")


def amp_dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def compute_forward_batch(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    components: TrainComponents,
    x0: torch.Tensor,
    y: object,
    use_label_cond: bool,
) -> ForwardBatch:
    x0 = x0.to(runtime.device, non_blocking=True)
    bsz = x0.shape[0]
    t = components.time_sampler.sample(bsz, device=runtime.device)
    cond = y.to(runtime.device, non_blocking=True) if (use_label_cond and isinstance(y, torch.Tensor)) else None
    if cond is not None:
        cond = apply_classifier_free_label_dropout(
            cond,
            null_class_id=int(model_ctx.model_cfg.num_classes),
            drop_prob=float(cfg["train"].get("class_dropout_prob", 0.0)),
        )

    if components.augment_mode == "data_only":
        x0 = components.augment.apply(x0, components.augment.sample_params(bsz, device=runtime.device))
    fb = components.process.forward_sample(x0=x0, t=t)

    with torch.autocast(
        device_type=runtime.device.type,
        dtype=amp_dtype_for_precision(model_ctx.precision),
        enabled=(model_ctx.precision in {"bf16", "fp16"}),
    ):
        out = model(fb.xt, fb.t, cond=cond)
        loss_dict = components.loss_fn(components.process, fb, out)
        with torch.no_grad():
            p = components.process
            unweighted = {
                "eps": regress("mse", p.eps_from_output(fb.xt, fb.t, out, aux=fb.aux), p.eps_target(fb)).mean(),
                "x0":  regress("mse", p.x0_from_output(fb.xt, fb.t, out, aux=fb.aux), p.x0_target(fb)).mean(),
                "v":   regress("mse", p.v_from_output(fb.xt, fb.t, out, aux=fb.aux), p.v_target(fb)).mean(),
            }

    return ForwardBatch(
        loss=loss_dict["total"],
        loss_by_target=unweighted,
        batch_size=bsz,
        cond=cond,
        fb=fb,
        out=out,
    )


def backward_loss(loss: torch.Tensor, *, current_accum_steps: int, scaler) -> None:
    loss_for_backward = loss / float(current_accum_steps)
    if scaler is not None:
        scaler.scale(loss_for_backward).backward()
    else:
        loss_for_backward.backward()


def should_step_optimizer(micro_step: int, loop_cfg: LoopConfig) -> bool:
    return ((micro_step + 1) % loop_cfg.gradient_accumulation_steps == 0) or (
        micro_step + 1 == loop_cfg.micro_steps_per_epoch
    )


def step_optimizer(model: torch.nn.Module, components: TrainComponents, grad_clip: float):
    if components.scaler is not None:
        components.scaler.unscale_(components.optimizer)
        grad_norm = clip_grad_norm(model, grad_clip) if grad_clip > 0 else clip_grad_norm(model, float("inf"))
        components.scaler.step(components.optimizer)
        components.scaler.update()
        return grad_norm

    grad_norm = clip_grad_norm(model, grad_clip) if grad_clip > 0 else clip_grad_norm(model, float("inf"))
    components.optimizer.step()
    return grad_norm


def update_train_meters(
    meters: MetricLogger,
    fwd: ForwardBatch,
    *,
    lr: float,
    iter_time: float,
    world_size: int,
    grad_norm=None,
) -> None:
    throughput = fwd.batch_size * world_size / max(iter_time, 1e-6)
    meters.update(
        loss=float(fwd.loss.detach().item()),
        lr=float(lr),
        iter_s=float(iter_time),
        img_s=float(throughput),
    )
    for target, val in fwd.loss_by_target.items():
        meters.update(**{f"loss_{target}": float(val.detach().item())})
    if grad_norm is not None:
        grad_norm_value = float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        meters.update(grad_norm=grad_norm_value)


def log_training_step(
    *,
    runtime: RuntimeContext,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    meters: MetricLogger,
    tbin_stats: TimeBinAccumulator,
    epoch: int,
    micro_step: int,
    current_accum_steps: int,
) -> None:
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

    kv["summary"] = tbin_stats.summary(is_distributed=runtime.is_distributed)
    runtime.logger.log_kv(resume.global_step, kv, total_steps=loop_cfg.total_steps)
    tbin_stats.reset()


def run_sampling_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    model_ctx: ModelContext,
    components: TrainComponents,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    cond: torch.Tensor | None,
    use_label_cond: bool,
) -> None:
    if loop_cfg.sample_every <= 0 or (resume.global_step % loop_cfg.sample_every != 0):
        return

    should_run_sample = (not loop_cfg.sample_rank0_only) or runtime.is_main
    # FSDP forward is collective; run on all ranks when sharded.
    if model_ctx.use_fsdp and runtime.is_distributed:
        should_run_sample = True

    if should_run_sample:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            if components.ema is not None:
                components.ema.store(model)
                components.ema.copy_to(model)
            sample_num = int(cfg["sample"].get("num", 16))
            sample_cond = build_sample_cond(
                cfg,
                sample_num=sample_num,
                device=runtime.device,
                batch_cond=cond if use_label_cond else None,
            )
            null_cond = build_null_class_cond(cfg, sample_num=sample_num, device=runtime.device)
            guidance_scale = float(cfg["sample"].get("guidance_scale", 1.0))
            sample_sampler = cfg["sample"].get("sampler", None)
            if isinstance(sample_sampler, str) and sample_sampler.lower() == "auto":
                sample_sampler = None
            result = components.process.sample(
                model=model,
                steps=int(cfg["sample"].get("steps", 50)),
                shape=(sample_num, model_ctx.model_cfg.out_channels, model_ctx.model_cfg.image_size, model_ctx.model_cfg.image_size),
                device=runtime.device,
                dtype=torch.float32,
                return_trace=True,
                cond=sample_cond,
                null_cond=null_cond,
                guidance_scale=guidance_scale,
                sampler=sample_sampler,
                posterior_noise_scale=cfg["sample"].get("posterior_noise_scale", None),
            )
            if components.ema is not None:
                components.ema.restore(model)

        if runtime.is_main:
            sample_dir = os.path.join(runtime.out_dir, "samples")
            save_trace_large_images(result.get("trace", []), sample_dir, f"step_{resume.global_step:08d}")
            if bool(cfg["sample"].get("save_trace", False)) and "trace" in result:
                trace_path = os.path.join(sample_dir, f"step_{resume.global_step:08d}_trace.pt")
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                torch.save(result["trace"], trace_path)

        if was_training:
            model.train()

    if runtime.is_distributed:
        dist_utils.barrier()


def save_checkpoint_if_due(
    *,
    cfg: dict,
    model: torch.nn.Module,
    runtime: RuntimeContext,
    components: TrainComponents,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    epoch: int,
) -> None:
    if loop_cfg.save_every <= 0 or (resume.global_step % loop_cfg.save_every != 0):
        return

    ckpt_path = os.path.join(runtime.out_dir, "checkpoints", f"ckpt_step_{resume.global_step:08d}.pt")
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=components.optimizer,
        scaler=components.scaler,
        ema=components.ema,
        step=resume.global_step,
        epoch=epoch,
        config=cfg,
        is_main=runtime.is_main,
        mode=resume.ckpt_mode,
    )
    if runtime.is_distributed:
        dist_utils.barrier()


def train(cfg: dict):
    runtime = init_runtime(cfg)
    data_ctx = build_data_context(cfg, runtime)
    model_ctx = build_model_context(cfg, runtime)
    components = build_train_components(cfg, model_ctx, runtime)
    resume = load_resume_state(cfg, model_ctx, components, runtime)
    meters = MetricLogger(window_size=int(cfg["logging"].get("window_size", 20)))
    loop_cfg = build_loop_config(cfg, data_ctx.loader, runtime.distributed_cfg)
    if runtime.is_main:
        log_loop_config(runtime.logger, loop_cfg)
    tbin_stats = TimeBinAccumulator(
        num_bins=loop_cfg.tbin_count,
        device=runtime.device,
    )

    model = model_ctx.model
    model.train()
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

            if update_step:
                step_lr = float(loop_cfg.lr_for_step(resume.global_step))
                for pg in components.optimizer.param_groups:
                    pg["lr"] = step_lr
                components.optimizer.zero_grad(set_to_none=True)

            fwd = compute_forward_batch(
                cfg=cfg,
                model=model,
                runtime=runtime,
                model_ctx=model_ctx,
                components=components,
                x0=x0,
                y=y,
                use_label_cond=use_label_cond,
            )
            backward_loss(fwd.loss, current_accum_steps=current_accum_steps, scaler=components.scaler)

            did_optimizer_step = should_step_optimizer(micro_step, loop_cfg)
            grad_norm = None
            if did_optimizer_step:
                effective_clip = 0.0 if resume.global_step < 10000 else loop_cfg.grad_clip
                grad_norm = step_optimizer(model, components, effective_clip)

            if did_optimizer_step and components.ema is not None:
                components.ema.update(model)

            tbin_stats.update(
                schedule=components.schedule,
                process=components.process,
                loss_fn=components.loss_fn,
                fb=fwd.fb,
                out=fwd.out,
            )

            iter_time = time.time() - iter_start
            iter_start = time.time()
            update_train_meters(
                meters,
                fwd,
                lr=float(components.optimizer.param_groups[0]["lr"]),
                iter_time=iter_time,
                world_size=runtime.world_size,
                grad_norm=grad_norm if did_optimizer_step else None,
            )

            if not did_optimizer_step:
                continue

            resume.global_step += 1
            resume.run_step += 1

            log_training_step(
                runtime=runtime,
                loop_cfg=loop_cfg,
                resume=resume,
                meters=meters,
                tbin_stats=tbin_stats,
                epoch=epoch,
                micro_step=micro_step,
                current_accum_steps=current_accum_steps,
            )
            run_sampling_if_due(
                cfg=cfg,
                model=model,
                runtime=runtime,
                model_ctx=model_ctx,
                components=components,
                loop_cfg=loop_cfg,
                resume=resume,
                cond=fwd.cond,
                use_label_cond=use_label_cond,
            )
            save_checkpoint_if_due(
                cfg=cfg,
                model=model,
                runtime=runtime,
                components=components,
                loop_cfg=loop_cfg,
                resume=resume,
                epoch=epoch,
            )

    runtime.logger.close()


def _apply_set_overrides(cfg: dict, overrides: list[str]) -> dict:
    def _cast(v: str):
        if v.lower() == "true":  return True
        if v.lower() == "false": return False
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
        return v

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set requires key=value format, got: {item!r}")
        key_path, _, raw_val = item.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _cast(raw_val.strip())
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="x0loop/configs/default.yaml")
    parser.add_argument("--runtime-config", type=str, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_merged_config(args.config, args.runtime_config, resolve_logging=False)
    _apply_set_overrides(cfg, args.overrides)
    train(cfg)
