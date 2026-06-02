from __future__ import annotations

import copy
import os
from pathlib import Path
import re
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


def _path_component(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", text).strip("-") or "unknown"


def _loss_name(cfg: dict) -> str:
    loss_cfg = cfg.get("loss", {}) or {}
    terms = loss_cfg.get("terms")
    if isinstance(terms, list):
        targets = [term.get("target", "unknown") for term in terms if isinstance(term, dict)]
    else:
        targets = [loss_cfg.get("target", "unknown")]
    return "-".join(dict.fromkeys(_path_component(target) for target in targets)) or "unknown"


def _sampler_name(cfg: dict) -> str:
    process_cfg = cfg.get("process", {}) or {}
    sample_cfg = cfg.get("sample", {}) or {}
    process_name = _path_component(process_cfg.get("name", "diffusion"))
    sampler = str(sample_cfg.get("sampler", "auto")).lower()
    if sampler in {"", "auto"}:
        sampler = str(process_cfg.get("sampler", "euler" if process_name == "flow" else "ddim")).lower()
    if process_name == "flow" and sampler == "ddim":
        sampler = "euler"
    return _path_component(sampler)


def _automatic_run_base(cfg: dict) -> Path:
    dataset = _path_component((cfg.get("dataset", {}) or {}).get("name", "unknown"))
    process_cfg = cfg.get("process", {}) or {}
    process = _path_component(process_cfg.get("name", "diffusion"))
    model = _path_component((cfg.get("model", {}) or {}).get("name", "dit"))
    output_target = _path_component(process_cfg.get("output_target", "eps"))
    experiment = f"{output_target}target_{_loss_name(cfg)}loss_{_sampler_name(cfg)}"
    return Path("runs") / dataset / process / model / experiment


def resolve_logging_output_dir(cfg: dict, timestamp: str | None = None) -> None:
    logging_cfg = cfg.get("logging")
    if not isinstance(logging_cfg, dict):
        return

    if logging_cfg.get("out_dir"):
        return

    if not timestamp:
        timestamp = os.environ.get("X0LOOP_RUN_TIMESTAMP")
    if not timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        os.environ["X0LOOP_RUN_TIMESTAMP"] = timestamp
    else:
        os.environ["X0LOOP_RUN_TIMESTAMP"] = timestamp

    logging_cfg["out_dir"] = str(_automatic_run_base(cfg) / timestamp)


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
