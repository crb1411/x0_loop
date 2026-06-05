from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from x0loop.utils.logger import Logger


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
    eval_loader: DataLoader | None = None


@dataclass
class ModelContext:
    model: torch.nn.Module
    model_cfg: object
    use_fsdp: bool
    fsdp_mode: str
    precision: str


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
    loss_by_target: dict
    batch_size: int
    cond: torch.Tensor | None
    fb: object
    out: torch.Tensor
    extra_metrics: dict | None = None
