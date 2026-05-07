from __future__ import annotations

import copy
import os
from pathlib import Path
import time

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


def resolve_logging_output_dir(cfg: dict, timestamp: str | None = None) -> None:
    logging_cfg = cfg.get("logging")
    if not isinstance(logging_cfg, dict):
        return

    if logging_cfg.get("out_dir"):
        return

    out_dir_base = logging_cfg.get("out_dir_base")
    if not out_dir_base:
        return

    if not timestamp:
        timestamp = os.environ.get("X0LOOP_RUN_TIMESTAMP")
    if not timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        os.environ["X0LOOP_RUN_TIMESTAMP"] = timestamp
    else:
        os.environ["X0LOOP_RUN_TIMESTAMP"] = timestamp

    stage = str(logging_cfg.get("stage", "train"))
    logging_cfg["out_dir"] = str(Path(out_dir_base) / f"{timestamp}_{stage}")


def load_merged_config(config_path: str, runtime_config_path: str | None = None, *, resolve_logging: bool = True) -> dict:
    cfg = _load_yaml(config_path)
    runtime_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG
    if runtime_path and Path(runtime_path).exists():
        runtime_cfg = _load_yaml(runtime_path)
        cfg = _deep_merge_dict(cfg, runtime_cfg)
    if resolve_logging:
        resolve_logging_output_dir(cfg)
    cfg["_config_path"] = config_path
    cfg["_runtime_config_path"] = runtime_path
    return cfg


def dump_resolved_config(cfg: dict, out_dir: str, name: str = "resolved_config.yaml") -> str:
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(path)
