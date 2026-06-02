from __future__ import annotations

import argparse

from x0loop.core.config import DEFAULT_RUNTIME_CONFIG, load_merged_config
from x0loop.training.engine import train
from x0loop.training.factories import build_process, build_schedule
from x0loop.training.sampling import build_null_class_cond, build_sample_cond, save_sample_grid, save_trace_large_images

__all__ = [
    "build_null_class_cond",
    "build_process",
    "build_sample_cond",
    "build_schedule",
    "save_sample_grid",
    "save_trace_large_images",
    "train",
]


def _apply_set_overrides(cfg: dict, overrides: list[str]) -> dict:
    def _cast(value: str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set requires key=value format, got: {item!r}")
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _cast(raw_value.strip())
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="x0loop/configs/default.yaml")
    parser.add_argument("--runtime-config", type=str, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_merged_config(args.config, args.runtime_config, resolve_logging=False)
    _apply_set_overrides(cfg, args.overrides)
    train(cfg)
