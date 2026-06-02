from __future__ import annotations

import os

import torch

from x0loop.training.context import LoopConfig, ResumeState, RuntimeContext
from x0loop.utils import dist as dist_utils
from x0loop.utils.checkpoint import save_checkpoint


def save_checkpoint_if_due(
    *,
    cfg: dict,
    denoiser: torch.nn.Module,
    runtime: RuntimeContext,
    optimizer,
    scaler,
    ema,
    loop_cfg: LoopConfig,
    resume: ResumeState,
    epoch: int,
) -> None:
    if loop_cfg.save_every <= 0 or (resume.global_step % loop_cfg.save_every != 0):
        return

    ckpt_path = os.path.join(runtime.out_dir, "checkpoints", f"ckpt_step_{resume.global_step:08d}.pt")
    save_checkpoint(
        path=ckpt_path,
        model=denoiser,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        step=resume.global_step,
        epoch=epoch,
        config=cfg,
        is_main=runtime.is_main,
        mode=resume.ckpt_mode,
    )
    if runtime.is_distributed:
        dist_utils.barrier()
