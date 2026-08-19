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
    def _fmt_int(value, width: int | None = None) -> str:
        try:
            n = int(round(float(value)))
            return f"{n:0{width}d}" if width is not None and width > 0 else str(n)
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
    def _format_training_preview(cls, kv: dict, *, total_steps: int | None = None) -> str:
        def val(key: str, default: float = 0.0):
            return kv.get(key, default)

        total_loss = val("loss")
        fresh_scale = val("clean/fresh_scale", 1.0)
        fresh_loss = val("fresh/loss", total_loss)
        step_width = len(str(int(total_steps))) if total_steps is not None and total_steps > 0 else 0
        if "clean/loss_aux" in kv:
            aux_label = "aux_x0_mse" if val("clean/aux_target_x0", 0.0) >= 0.5 else "aux_velocity_mse"
            loss_part = (
                f"(loss) {cls._fmt_sig(total_loss)} = "
                f"(fresh) {cls._fmt_sig(val('fresh/loss_contrib', fresh_loss))} + "
                f"(aux_scale) {cls._fmt_sig(val('clean/aux_scale'))} * "
                f"({aux_label}) {cls._fmt_sig(val('clean/loss_aux'))}"
            )
        else:
            bank_scale = val("clean/bank_scale", 0.0)
            bank_weight = val("clean/loss_bank_weight", 0.0)
            bank_loss = val("clean/loss_bank", 0.0)
            loss_part = (
                f"(loss) {cls._fmt_sig(total_loss)} = "
                f"(fresh_scale) {cls._fmt_sig(fresh_scale)} * "
                f"(weighted_fresh_loss) {cls._fmt_sig(fresh_loss)} + "
                f"(bank_scale) {cls._fmt_sig(bank_scale)} * "
                f"(bank_weight) {cls._fmt_sig(bank_weight)} * "
                f"(bank_x0_mse) {cls._fmt_sig(bank_loss)}"
            )
        parts: list[str] = [loss_part]
        mid_tbin = cls._format_mid_tbin(kv.get("summary"))
        if mid_tbin is not None:
            parts.append(mid_tbin)

        base = []
        if "lr" in kv:
            base.append(f"lr={cls._fmt_sci(kv['lr'])}")
        if "iter_s" in kv:
            base.append(f"iter_s={cls._fmt_num(kv['iter_s']):>7}")
        if "img_s" in kv:
            base.append(f"img_s={cls._fmt_num(kv['img_s']):>9}")
        if "grad_norm" in kv:
            base.append(f"grad_norm={cls._fmt_num(kv['grad_norm']):>8}")
        if "epoch" in kv:
            base.append(f"epoch={cls._fmt_int(kv['epoch'], width=4)}")
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
        clean_int_values = [
            kv[key]
            for key in ("clean/bank_size", "clean/bank_n", "clean/fresh_n", "clean/aux_n", "clean/warmup_left", "clean/fresh_add_n", "clean/bank_add_n")
            if key in kv
        ]
        clean_int_width = max(5, *(len(cls._fmt_int(v)) for v in clean_int_values)) if clean_int_values else 5
        for key, label in (
            ("clean/bank_size", "bank_size"),
            ("clean/bank_n", "bank_n"),
            ("clean/fresh_n", "fresh_n"),
            ("clean/aux_n", "aux_n"),
            ("clean/aux_output_grad_ratio", "aux_grad_ratio"),
            ("clean/solver_index", "solver_index"),
            ("clean/depth", "depth"),
            ("clean/bank_prob", "bank_prob"),
            ("clean/t_bank", "t_bank"),
            ("clean/t1", "t1"),
            ("clean/fresh_add_n", "fresh_add_n"),
            ("clean/bank_add_n", "bank_add_n"),
            ("clean/warmup_left", "warmup_left"),
            ("clean/bank_age", "bank_age"),
        ):
            if key in kv:
                if key in {"clean/bank_size", "clean/bank_n", "clean/fresh_n", "clean/aux_n", "clean/warmup_left", "clean/fresh_add_n", "clean/bank_add_n"}:
                    clean.append(f"{label}={cls._fmt_int(kv[key], width=clean_int_width)}")
                else:
                    clean.append(f"{label}={cls._fmt_num(kv[key])}")
        if clean:
            parts.append(f"(clean) {' '.join(clean)}")

        progress = []
        if "progress_pct" in kv:
            try:
                progress.append(f"progress_pct={float(kv['progress_pct']):7.5f}%")
            except (TypeError, ValueError):
                progress.append(f"progress_pct={kv['progress_pct']}")
        for key, width in (("elapsed", 6), ("eta_train", 9), ("eta_geneval", 9), ("eta_total", 9), ("total_est", 9)):
            if key not in kv:
                continue
            value = kv[key]
            if isinstance(value, (float, int)):
                value = cls._fmt_num(value)
            progress.append(f"{key}={str(value):>{width}}")
        if progress:
            parts.append(f"(progress) {' '.join(progress)}")

        meta = []
        if "gpu_mem_gb" in kv:
            meta.append(f"gpu_mem_gb={cls._fmt_num(kv['gpu_mem_gb'])}")
        if "micro_step" in kv:
            meta.append(f"micro_step={cls._fmt_int(kv['micro_step'], width=step_width or 5)}")
        if "accumulation_steps" in kv:
            meta.append(f"accumulation_steps={cls._fmt_int(kv['accumulation_steps'], width=2)}")
        if meta:
            parts.append(f"(meta) {' '.join(meta)}")

        if "summary" in kv:
            parts.append(f"(summary) {kv['summary']}")
        return " | ".join(parts)

    @classmethod
    def _format_default_preview(cls, kv: dict) -> str:
        return " ".join([f"{k}={cls._fmt_num(v)}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in kv.items()])

    @classmethod
    def _format_preview(cls, kv: dict, *, total_steps: int | None = None) -> str:
        if "loss" in kv and "fresh/loss" in kv:
            return cls._format_training_preview(kv, total_steps=total_steps)
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
        preview = self._format_preview(kv, total_steps=total_steps)
        if total_steps is not None and total_steps > 0:
            width = len(str(int(total_steps)))
            self.py_logger.info("[step %0*d/%d] %s", width, step, int(total_steps), preview)
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
        self._pending_tensor_updates: list[tuple[str, torch.Tensor, int]] = []
        self._last_reduce_ts = 0.0

    def _update_number(self, key: str, value: float, *, n: int = 1) -> None:
        if key not in self.meters:
            self.meters[key] = SmoothedValue(window_size=self.window_size)
        self.meters[key].update(float(value), n=n)

    def update(self, *, n: int = 1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                value = v.detach()
                if value.numel() != 1:
                    raise ValueError(f"MetricLogger tensor metric {k!r} must be scalar, got shape={tuple(value.shape)}")
                if value.device.type != "cpu":
                    self._pending_tensor_updates.append((k, value.reshape(()), int(n)))
                    continue
                v = float(value.item())
            if not isinstance(v, (float, int)):
                continue
            self._update_number(k, float(v), n=int(n))

    def flush_pending(self) -> None:
        """Materialize all queued device scalars with one transfer per device.

        Training queues tensor-valued diagnostics every step and calls this at
        logging cadence. Packing first avoids a train-loop synchronization for
        every individual ``Tensor.item()`` while preserving every meter sample.
        """
        if not self._pending_tensor_updates:
            return
        pending = self._pending_tensor_updates
        self._pending_tensor_updates = []
        grouped: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
        for index, (_, value, _) in enumerate(pending):
            grouped.setdefault(value.device, []).append((index, value))
        resolved: list[float | None] = [None] * len(pending)
        for values in grouped.values():
            packed = torch.stack([value.float() for _, value in values])
            host_values = packed.cpu().tolist()
            for (index, _), host_value in zip(values, host_values, strict=True):
                resolved[index] = float(host_value)
        for (key, _, n), value in zip(pending, resolved, strict=True):
            assert value is not None
            self._update_number(key, value, n=n)

    def reduce_distributed(self):
        self.flush_pending()
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
        self.flush_pending()
        out = {}
        for name, meter in self.meters.items():
            if meter.synced_global_avg is not None:
                out[name] = meter.synced_global_avg
            elif meter.synced_avg is not None:
                out[name] = meter.synced_avg
            else:
                out[name] = meter.avg
        return out
