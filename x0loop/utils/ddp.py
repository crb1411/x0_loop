from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel


def wrap_ddp(
    model: torch.nn.Module,
    *,
    device: torch.device,
    local_rank: int,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = False,
    static_graph: bool = False,
) -> DistributedDataParallel:
    if device.type == "cuda":
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
    else:
        wrapped = DistributedDataParallel(
            model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
    setattr(wrapped, "_x0loop_ddp_enabled", True)
    return wrapped
