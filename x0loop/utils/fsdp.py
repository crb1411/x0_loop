from __future__ import annotations

from dataclasses import dataclass
import functools

import torch


@dataclass
class FSDPMeta:
    enabled: bool
    mode: str

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float):
        if max_norm is None or max_norm <= 0:
            return None
        if self.enabled and self.mode == "fsdp1":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            return FSDP.clip_grad_norm_(model, max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    _HAS_FSDP1 = True
except Exception:
    _HAS_FSDP1 = False


def _dtype_from_precision(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _mark_fsdp_mode(model: torch.nn.Module, mode: str) -> None:
    setattr(model, "_x0loop_fsdp_mode", mode)
    setattr(model, "_x0loop_fsdp_enabled", mode in {"fsdp1", "fsdp2"})


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
    del use_compile

    target_device = torch.device("cuda", device_id) if torch.cuda.is_available() else torch.device("cpu")
    if activation_ckpt:
        _apply_activation_checkpointing(model)

    # Prefer composable FSDP2 if available.
    try:
        from torch.distributed._composable.fsdp import fully_shard

        for mod in model.modules():
            if mod.__class__.__name__.endswith("DiTBlock"):
                fully_shard(mod)
        fully_shard(model)
        model = model.to(target_device)
        _mark_fsdp_mode(model, "fsdp2")
        return model, FSDPMeta(enabled=True, mode="fsdp2")
    except Exception:
        pass

    if not _HAS_FSDP1:
        model = model.to(target_device)
        _mark_fsdp_mode(model, "none")
        return model, FSDPMeta(enabled=False, mode="none")

    dtype = _dtype_from_precision(precision)
    mp = None
    if mixed_precision:
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    # Auto-wrap blocks only when possible.
    dit_block_cls = None
    for mod in model.modules():
        if mod.__class__.__name__.endswith("DiTBlock"):
            dit_block_cls = mod.__class__
            break

    wrap_policy = None
    if dit_block_cls is not None:
        wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={dit_block_cls})

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=target_device if target_device.type == "cuda" else None,
        use_orig_params=True,
    )
    _mark_fsdp_mode(model, "fsdp1")
    return model, FSDPMeta(enabled=True, mode="fsdp1")
