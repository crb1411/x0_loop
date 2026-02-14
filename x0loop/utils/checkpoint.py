from __future__ import annotations

import os

import torch


def _strip_state_dict_prefix(state_dict: dict, prefix: str) -> dict:
    plen = len(prefix)
    out = {}
    for k, v in state_dict.items():
        nk = k[plen:] if k.startswith(prefix) else k
        out[nk] = v
    return out


def _candidate_state_dicts(state_dict: dict) -> list[tuple[str, dict]]:
    # Try common wrapper prefixes from compile/DDP/FSDP stacks.
    cands: list[tuple[str, dict]] = [("none", state_dict)]
    prefixes = [
        "_orig_mod.",
        "module.",
        "_fsdp_wrapped_module.",
        "model.",
    ]
    seen = {tuple(state_dict.keys())}
    for p in prefixes:
        sd = _strip_state_dict_prefix(state_dict, p)
        key_sig = tuple(sd.keys())
        if key_sig not in seen:
            cands.append((p, sd))
            seen.add(key_sig)
    return cands


def _load_model_state_with_fallback(model, state_dict: dict, strict: bool = True) -> dict:
    last_err: Exception | None = None
    candidates = _candidate_state_dicts(state_dict)
    for prefix, sd in candidates:
        try:
            incompatible = model.load_state_dict(sd, strict=strict)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            return {
                "strict": bool(strict),
                "prefix": prefix,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            }
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return {"strict": bool(strict), "prefix": "none", "missing_keys": [], "unexpected_keys": []}


def _get_fsdp_mode(model) -> str:
    mode = getattr(model, "_x0loop_fsdp_mode", None)
    if isinstance(mode, str) and mode in {"fsdp1", "fsdp2", "none"}:
        return mode
    if model.__class__.__name__ == "FullyShardedDataParallel":
        return "fsdp1"
    return "none"


def _maybe_fsdp2_full_state_dict(model):
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    except Exception:
        pass
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        model_state, _ = get_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        return model_state
    except Exception:
        return model.state_dict()


def save_checkpoint(
    path: str,
    model,
    optimizer,
    *,
    scaler=None,
    ema=None,
    step: int,
    epoch: int,
    config: dict | None = None,
    is_main: bool = True,
    mode: str = "full",
):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fsdp_mode = _get_fsdp_mode(model)
    use_fsdp = fsdp_mode in {"fsdp1", "fsdp2"}
    if fsdp_mode == "fsdp1":
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                ShardedStateDictConfig,
                StateDictType,
            )
        except Exception:
            use_fsdp = False

    if use_fsdp and mode == "sharded":
        if fsdp_mode == "fsdp1":
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
                model_state = model.state_dict()
        else:
            model_state = model.state_dict()
        ckpt = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "config": config or {},
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        shard_path = path.replace(".pt", f"_rank{rank:04d}.pt")
        torch.save(ckpt, shard_path)
        return

    if use_fsdp and mode == "full":
        if fsdp_mode == "fsdp1":
            if not is_main:
                return
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state = model.state_dict()
        else:
            model_state = _maybe_fsdp2_full_state_dict(model)
            if not is_main:
                return
    else:
        if not is_main:
            return
        model_state = model.state_dict()

    ckpt = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": config or {},
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
    ema=None,
    map_location="cpu",
    mode: str = "full",
    strict: bool = True,
):
    rank = int(os.environ.get("RANK", "0"))
    load_path = path
    shard_path = None
    if mode == "sharded":
        shard_path = path.replace(".pt", f"_rank{rank:04d}.pt")
        if os.path.exists(shard_path):
            load_path = shard_path
    ckpt = torch.load(load_path, map_location=map_location)
    try:
        load_info = _load_model_state_with_fallback(model, ckpt["model"], strict=strict)
    except Exception as first_err:
        # Fallback path: if sharded file fails but full checkpoint exists, retry full.
        if mode == "sharded" and shard_path is not None and load_path == shard_path and os.path.exists(path):
            ckpt = torch.load(path, map_location=map_location)
            load_info = _load_model_state_with_fallback(model, ckpt["model"], strict=strict)
        else:
            raise first_err
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    ckpt["_load_info"] = load_info
    return ckpt
