from __future__ import annotations

import functools

import torch


def get_fsdp_mode(model: torch.nn.Module) -> str:
    return str(getattr(model, "_x0loop_fsdp_mode", "none"))


def clip_grad_norm(model: torch.nn.Module, max_norm: float):
    if max_norm is None or max_norm <= 0:
        return None
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _mark_fsdp_mode(model: torch.nn.Module, mode: str) -> None:
    setattr(model, "_x0loop_fsdp_mode", mode)
    setattr(model, "_x0loop_fsdp_enabled", mode == "fsdp2")


def _apply_activation_checkpointing(model: torch.nn.Module) -> None:
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except Exception:
        return

    wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule: torch.nn.Module) -> bool:
        return submodule.__class__.__name__.endswith("DiTBlock")

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


def wrap_fsdp2(
    model,
    *,
    mixed_precision: bool,
    precision: str,
    use_compile: bool,
    activation_ckpt: bool,
    device_id: int,
):
    del mixed_precision, precision, use_compile

    target_device = torch.device("cuda", device_id) if torch.cuda.is_available() else torch.device("cpu")
    if activation_ckpt:
        _apply_activation_checkpointing(model)

    try:
        from torch.distributed._composable.fsdp import fully_shard
    except Exception as e:
        raise RuntimeError("FSDP2 is required when distributed.fsdp=true, but it is not available.") from e

    try:
        for mod in model.modules():
            if mod.__class__.__name__.endswith("DiTBlock"):
                fully_shard(mod)
        fully_shard(model)
    except Exception as e:
        raise RuntimeError("Failed to wrap model with FSDP2.") from e

    model = model.to(target_device)
    _mark_fsdp_mode(model, "fsdp2")
    return model, "fsdp2"
