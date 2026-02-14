from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _build_python_logger(name: str, *, out_dir: str, is_main: bool, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate handlers if logger is re-initialized.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if is_main:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    log_file = os.path.join(log_dir, "log.txt")
    if not is_main:
        log_file = f"{log_file}.rank{rank}"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class Logger:
    def __init__(self, out_dir: str, *, is_main: bool, use_tb: bool, flush_secs: int = 5):
        self.out_dir = out_dir
        self.is_main = is_main
        self.use_tb = bool(use_tb and SummaryWriter is not None)
        self.run_timestamp = os.environ.get("X0LOOP_RUN_TIMESTAMP", time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        self.metrics_path = ""
        self.tb = None
        self.fp = None
        self.py_logger = _build_python_logger("x0loop", out_dir=out_dir, is_main=is_main)

        if self.is_main:
            os.makedirs(out_dir, exist_ok=True)
            metrics_name = f"metrics_{self.run_timestamp}.jsonl"
            self.metrics_path = os.path.join(out_dir, metrics_name)
            if os.path.exists(self.metrics_path):
                self.metrics_path = os.path.join(out_dir, f"metrics_{self.run_timestamp}_{os.getpid()}.jsonl")
            self.fp = open(self.metrics_path, "w", encoding="utf-8")
            if self.use_tb:
                self.tb = SummaryWriter(log_dir=out_dir, flush_secs=flush_secs)
            self.py_logger.info("metrics_file=%s", self.metrics_path)

    @staticmethod
    def _now_str() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def log_text(self, msg: str):
        if self.is_main:
            self.py_logger.info("%s", msg)

    def log_kv(self, step: int, kv: dict, total_steps: int | None = None):
        if not self.is_main:
            return
        ts = self._now_str()
        data = {"step": int(step), "time": ts, **kv}
        if total_steps is not None:
            data["total_steps"] = int(total_steps)
        line = json.dumps(data, ensure_ascii=True)
        self.fp.write(line + "\n")
        self.fp.flush()
        for k, v in kv.items():
            if self.tb is not None and isinstance(v, (float, int)):
                self.tb.add_scalar(k, float(v), step)
        preview = " ".join([f"{k}={v:.5g}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in kv.items()])
        if total_steps is not None and total_steps > 0:
            self.py_logger.info("[step %d/%d] %s", step, int(total_steps), preview)
        else:
            self.py_logger.info("[step %d] %s", step, preview)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self.fp is not None:
            self.fp.close()
        for h in list(self.py_logger.handlers):
            try:
                h.flush()
                h.close()
            finally:
                self.py_logger.removeHandler(h)


class SmoothedValue:
    def __init__(self, window_size: int = 20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.synced_avg: float | None = None
        self.synced_global_avg: float | None = None

    def update(self, value: float, n: int = 1):
        self.deque.append(float(value))
        self.total += float(value) * n
        self.count += n
        self.synced_avg = None
        self.synced_global_avg = None

    @property
    def avg(self) -> float:
        if len(self.deque) == 0:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self) -> float:
        return self.total / max(1, self.count)


class MetricLogger:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.meters: dict[str, SmoothedValue] = {}
        self._last_reduce_ts = 0.0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = float(v.detach().item())
            if not isinstance(v, (float, int)):
                continue
            if k not in self.meters:
                self.meters[k] = SmoothedValue(window_size=self.window_size)
            self.meters[k].update(float(v), n=1)

    def reduce_distributed(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        world_size = torch.distributed.get_world_size()
        for meter in self.meters.values():
            avg_t = torch.tensor([meter.avg], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(avg_t, op=torch.distributed.ReduceOp.SUM)
            meter.synced_avg = float((avg_t / world_size)[0].item())

            t = torch.tensor([meter.total, meter.count], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            global_count = max(1.0, float(t[1].item()))
            meter.synced_global_avg = float(t[0].item() / global_count)
        self._last_reduce_ts = time.time()

    def get_log_dict(self) -> dict:
        out = {}
        for name, meter in self.meters.items():
            if meter.synced_global_avg is not None:
                out[name] = meter.synced_global_avg
            elif meter.synced_avg is not None:
                out[name] = meter.synced_avg
            else:
                out[name] = meter.avg
        return out
