from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


_DIST_INFO = {
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
    "is_distributed": False,
    "is_main": True,
}


def init_distributed(backend: str = "nccl") -> dict:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()

    _DIST_INFO.update(
        {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "is_distributed": is_distributed,
            "is_main": rank == 0,
        }
    )
    return dict(_DIST_INFO)


def is_main_process() -> bool:
    return bool(_DIST_INFO["is_main"])


def get_rank() -> int:
    return int(_DIST_INFO["rank"])


def get_world_size() -> int:
    return int(_DIST_INFO["world_size"])


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_reduce_mean(x):
    if isinstance(x, (float, int)):
        t = torch.tensor(float(x), device="cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()
        return float(t.item())

    if isinstance(x, torch.Tensor):
        y = x.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            y = y / dist.get_world_size()
        return y

    raise TypeError(f"Unsupported type for all_reduce_mean: {type(x)}")


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
