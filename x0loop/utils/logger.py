from __future__ import annotations

import json
import logging
import math
import os
import re
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

    @staticmethod
    def _fmt_num(value, digits: int = 5) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_sig(value, digits: int = 5) -> str:
        try:
            v = float(value)
            if not math.isfinite(v):
                return str(v)
            if v == 0.0:
                return "0." + ("0" * (digits - 1))
            decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(v)))))
            return f"{v:.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_sci(value, digits: int = 5) -> str:
        try:
            return f"{float(value):.{digits}e}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_int(value) -> str:
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_mid_tbin(summary) -> str | None:
        if not isinstance(summary, str):
            return None
        match = re.search(r"\[0\.50,0\.55\):\s*([^|]+)", summary)
        if match is None:
            return None
        fields = match.group(1).strip().rstrip()
        fields = fields.replace(", ", " ")
        return f"(tbin) [0.50,0.55) {fields}"

    @classmethod
    def _format_training_preview(cls, kv: dict) -> str:
        def val(key: str, default: float = 0.0):
            return kv.get(key, default)

        total_loss = val("loss")
        fresh_scale = val("clean/fresh_scale", 1.0)
        fresh_loss = val("fresh/loss", total_loss)
        bank_scale = val("clean/bank_scale", 0.0)
        bank_weight = val("clean/loss_bank_weight", 0.0)
        bank_loss = val("clean/loss_bank", 0.0)
        parts: list[str] = [
            f"(loss) {cls._fmt_sig(total_loss)} = "
            f"(fresh_scale) {cls._fmt_sig(fresh_scale)} * "
            f"(weighted_fresh_loss) {cls._fmt_sig(fresh_loss)} + "
            f"(bank_scale) {cls._fmt_sig(bank_scale)} * "
            f"(bank_weight) {cls._fmt_sig(bank_weight)} * "
            f"(bank_x0_mse) {cls._fmt_sig(bank_loss)}"
        ]
        mid_tbin = cls._format_mid_tbin(kv.get("summary"))
        if mid_tbin is not None:
            parts.append(mid_tbin)

        base = []
        for key in ("lr", "iter_s", "img_s", "grad_norm", "gpu_mem_gb"):
            if key not in kv:
                continue
            if key == "lr":
                base.append(f"{key}={cls._fmt_sci(kv[key])}")
            else:
                base.append(f"{key}={cls._fmt_num(kv[key])}")
        for key in ("epoch", "micro_step", "accumulation_steps"):
            if key in kv:
                base.append(f"{key}={cls._fmt_int(kv[key])}")
        if base:
            parts.append(" ".join(base))

        terminal_key = "loss_z" if "loss_z" in kv else "loss_eps" if "loss_eps" in kv else None
        diag = []
        if terminal_key is not None:
            diag.append(f"{terminal_key.removeprefix('loss_')}_mse={cls._fmt_sig(kv[terminal_key])}")
        if "loss_x0" in kv:
            diag.append(f"x0_mse={cls._fmt_sig(kv['loss_x0'])}")
        if "loss_v" in kv:
            diag.append(f"v_mse={cls._fmt_sig(kv['loss_v'])}")
        if diag:
            parts.append(f"(diag) {' '.join(diag)}")

        clean = []
        for key, label in (
            ("clean/bank_size", "bank_size"),
            ("clean/bank_n", "bank_n"),
            ("clean/fresh_n", "fresh_n"),
            ("clean/bank_prob", "bank_prob"),
            ("clean/warmup_left", "warmup_left"),
            ("clean/bank_age", "bank_age"),
        ):
            if key in kv:
                if key in {"clean/bank_size", "clean/bank_n", "clean/fresh_n", "clean/warmup_left"}:
                    clean.append(f"{label}={cls._fmt_int(kv[key])}")
                else:
                    clean.append(f"{label}={cls._fmt_num(kv[key])}")
        if clean:
            parts.append(f"(clean) {' '.join(clean)}")

        progress = []
        for key in ("progress_pct", "elapsed", "eta_train", "eta_geneval", "eta_total", "total_est"):
            if key in kv:
                value = kv[key]
                if key == "progress_pct" and isinstance(value, (float, int)):
                    value = f"{cls._fmt_num(value)}%"
                elif isinstance(value, (float, int)):
                    value = cls._fmt_num(value)
                progress.append(f"{key}={value}")
        if progress:
            parts.append(f"(progress) {' '.join(progress)}")

        if "summary" in kv:
            parts.append(f"(summary) {kv['summary']}")
        return " | ".join(parts)

    @classmethod
    def _format_default_preview(cls, kv: dict) -> str:
        return " ".join([f"{k}={cls._fmt_num(v)}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in kv.items()])

    @classmethod
    def _format_preview(cls, kv: dict) -> str:
        if "loss" in kv and "fresh/loss" in kv:
            return cls._format_training_preview(kv)
        return cls._format_default_preview(kv)

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
        preview = self._format_preview(kv)
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
