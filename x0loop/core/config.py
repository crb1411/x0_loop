from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_RUNTIME_CONFIG = "x0loop/configs/runtime/fsdp_checkpoint_compile.yaml"


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def load_merged_config(config_path: str, runtime_config_path: str | None = None) -> dict:
    cfg = _load_yaml(config_path)
    runtime_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG
    if runtime_path and Path(runtime_path).exists():
        runtime_cfg = _load_yaml(runtime_path)
        cfg = _deep_merge_dict(cfg, runtime_cfg)
    cfg["_config_path"] = config_path
    cfg["_runtime_config_path"] = runtime_path
    return cfg
