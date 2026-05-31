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
from x0loop.core.time_sampling import TimeSampler, build_time_sampler
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
    time_sampler: TimeSampler
    process: object
    loss_fn: CompositeLoss
    augment: object
    augment_mode: str
    optimizer: torch.optim.Optimizer
    scaler: object | None
    ema: EMA | None
