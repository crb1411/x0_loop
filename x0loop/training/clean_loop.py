from __future__ import annotations

from dataclasses import dataclass

import torch

from x0loop.losses.atomic import VALID_FORMULAS, normalize_loss_target


@dataclass(frozen=True)
class CleanLoopConfig:
    enabled: bool = False
    bank_size: int = 50000
    bank_prob: float = 0.25
    warmup_steps: int = 10000
    loss_bank_weight: float = 0.3
    time_constant: float = 0.5
    t_bank: float = 0.75
    bank_input_mode: str = "x0_hat_resample"
    t1_sampler: str = "local_uniform"
    t1_delta: float = 0.02
    t1_min: float = 1.0e-5
    t1_max_resample_attempts: int = 8
    bank_loss_target: str = "x0"
    bank_loss_formula: str = "mse"
    bank_loss_use_weight: bool = False
    storage_dtype: torch.dtype = torch.float16


def build_clean_loop_config(cfg: dict) -> CleanLoopConfig:
    raw = cfg.get("clean_loop", {}) or {}
    if not bool(raw.get("enabled", False)):
        return CleanLoopConfig(enabled=False)
    bank_size = int(raw.get("bank_size", raw.get("max_num", 50000)))
    bank_prob = float(raw.get("bank_prob", raw.get("p", 0.25)))
    warmup_steps = int(raw.get("warmup_steps", raw.get("warmupstep", 10000)))
    loss_bank_weight = float(raw.get("loss_bank_weight", 0.3))
    time_constant = float(raw.get("time_constant", cfg.get("model_conditioning", {}).get("time_constant", 0.5)))
    t_bank = float(raw.get("t_bank", raw.get("bank_add_min_t", 0.75)))
    bank_input_mode = str(raw.get("bank_input_mode", raw.get("bank_update_mode", "x0_hat_resample"))).lower()
    t1_sampler = str(raw.get("t1_sampler", "local_uniform")).lower()
    t1_delta = float(raw.get("t1_delta", raw.get("t1_max_delta", 1.0 / 50.0)))
    t1_min = float(raw.get("t1_min", 1.0e-5))
    t1_max_resample_attempts = int(raw.get("t1_max_resample_attempts", 8))
    bank_loss_raw = raw.get("bank_loss", {}) or {}
    bank_loss_target = normalize_loss_target(raw.get("bank_loss_target", bank_loss_raw.get("target", "x0")))
    bank_loss_formula = str(raw.get("bank_loss_formula", bank_loss_raw.get("formula", "mse"))).lower()
    bank_loss_use_weight = bool(raw.get("bank_loss_use_weight", bank_loss_raw.get("use_weight", False)))
    if bank_size <= 0:
        raise ValueError(f"clean_loop.bank_size must be > 0, got {bank_size}")
    if not (0.0 <= bank_prob <= 1.0):
        raise ValueError(f"clean_loop.bank_prob must be in [0, 1], got {bank_prob}")
    if warmup_steps < 0:
        raise ValueError(f"clean_loop.warmup_steps must be >= 0, got {warmup_steps}")
    if loss_bank_weight < 0.0:
        raise ValueError(f"clean_loop.loss_bank_weight must be >= 0, got {loss_bank_weight}")
    if not (0.0 <= t_bank <= 1.0):
        raise ValueError(f"clean_loop.t_bank must be in [0, 1], got {t_bank}")
    if bank_input_mode not in {"step", "x0_hat_resample", "resample"}:
        raise ValueError("clean_loop.bank_input_mode must be step | x0_hat_resample")
    if t1_sampler not in {"local_uniform", "resample_below_t", "time_sampler"}:
        raise ValueError("clean_loop.t1_sampler must be local_uniform | resample_below_t")
    if not (0.0 <= t1_min < 1.0):
        raise ValueError(f"clean_loop.t1_min must be in [0,1), got {t1_min}")
    if t1_delta <= 0.0:
        raise ValueError(f"clean_loop.t1_delta must be > 0, got {t1_delta}")
    if t1_max_resample_attempts <= 0:
        raise ValueError(f"clean_loop.t1_max_resample_attempts must be > 0, got {t1_max_resample_attempts}")
    if bank_loss_target == "mudata":
        raise ValueError("clean_loop bank loss does not support mudata target")
    if bank_loss_formula not in VALID_FORMULAS:
        raise ValueError(f"clean_loop.bank_loss_formula must be {' | '.join(sorted(VALID_FORMULAS))}, got {bank_loss_formula!r}")
    return CleanLoopConfig(
        enabled=True,
        bank_size=bank_size,
        bank_prob=bank_prob,
        warmup_steps=warmup_steps,
        loss_bank_weight=loss_bank_weight,
        time_constant=time_constant,
        t_bank=t_bank,
        bank_input_mode="x0_hat_resample" if bank_input_mode == "resample" else bank_input_mode,
        t1_sampler="resample_below_t" if t1_sampler == "time_sampler" else t1_sampler,
        t1_delta=t1_delta,
        t1_min=t1_min,
        t1_max_resample_attempts=t1_max_resample_attempts,
        bank_loss_target=bank_loss_target,
        bank_loss_formula=bank_loss_formula,
        bank_loss_use_weight=bank_loss_use_weight,
    )


class CleanLoopBank:
    """Local FIFO ring buffer for (x_t1, cond, x0, t1) tuples."""

    def __init__(self, cfg: CleanLoopConfig):
        self.cfg = cfg
        self.x_in: torch.Tensor | None = None
        self.x0: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.cond: torch.Tensor | None = None
        self.steps: torch.Tensor | None = None
        self.next_idx = 0
        self.count = 0

    def __len__(self) -> int:
        return self.count

    @property
    def capacity(self) -> int:
        return self.cfg.bank_size

    def _allocate(self, x_in: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None) -> None:
        shape = (self.capacity, *x_in.shape[1:])
        self.x_in = torch.empty(shape, dtype=self.cfg.storage_dtype, device="cpu")
        self.x0 = torch.empty(shape, dtype=self.cfg.storage_dtype, device="cpu")
        self.t = torch.empty((self.capacity,), dtype=torch.float32, device="cpu")
        self.steps = torch.empty((self.capacity,), dtype=torch.long, device="cpu")
        if cond is not None:
            self.cond = torch.empty((self.capacity, *cond.shape[1:]), dtype=cond.dtype, device="cpu")

    @torch.no_grad()
    def add(self, *, x_in: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None, step: int) -> None:
        if x_in.shape[0] == 0:
            return
        if self.x_in is None or self.x0 is None or self.t is None or self.steps is None:
            self._allocate(x_in, x0, t, cond)
        assert self.x_in is not None and self.x0 is not None and self.t is not None and self.steps is not None

        n = int(x_in.shape[0])
        x_in_cpu = x_in.detach().to(device="cpu", dtype=self.cfg.storage_dtype)
        x0_cpu = x0.detach().to(device="cpu", dtype=self.cfg.storage_dtype)
        t_cpu = t.detach().to(device="cpu", dtype=torch.float32).flatten()
        cond_cpu = cond.detach().to(device="cpu") if cond is not None else None

        for start in range(0, n, self.capacity):
            chunk_n = min(n - start, self.capacity)
            end = self.next_idx + chunk_n
            src = slice(start, start + chunk_n)
            if end <= self.capacity:
                dst = slice(self.next_idx, end)
                self.x_in[dst].copy_(x_in_cpu[src])
                self.x0[dst].copy_(x0_cpu[src])
                self.t[dst].copy_(t_cpu[src])
                self.steps[dst].fill_(int(step))
                if self.cond is not None and cond_cpu is not None:
                    self.cond[dst].copy_(cond_cpu[src])
            else:
                first = self.capacity - self.next_idx
                second = chunk_n - first
                self.add(
                    x_in=x_in_cpu[src][:first],
                    x0=x0_cpu[src][:first],
                    t=t_cpu[src][:first],
                    cond=cond_cpu[src][:first] if cond_cpu is not None else None,
                    step=step,
                )
                self.add(
                    x_in=x_in_cpu[src][first:first + second],
                    x0=x0_cpu[src][first:first + second],
                    t=t_cpu[src][first:first + second],
                    cond=cond_cpu[src][first:first + second] if cond_cpu is not None else None,
                    step=step,
                )
                continue
            self.next_idx = end % self.capacity
            self.count = min(self.capacity, self.count + chunk_n)

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= 0 or self.count <= 0 or self.x_in is None or self.x0 is None or self.t is None or self.steps is None:
            raise ValueError("CleanLoopBank.sample requires a non-empty bank and n > 0")
        n = min(int(n), self.count)
        idx = torch.randint(self.count, (n,), device="cpu")
        x_in = self.x_in[idx].to(device=device, dtype=dtype)
        x0 = self.x0[idx].to(device=device, dtype=dtype)
        t = self.t[idx].to(device=device)
        cond = self.cond[idx].to(device=device) if self.cond is not None else None
        steps = self.steps[idx].to(device=device)
        return x_in, cond, x0, t, steps


def sample_clean_loop_t1(
    *,
    cfg: CleanLoopConfig,
    t: torch.Tensor,
    time_sampler,
    device: torch.device,
) -> torch.Tensor:
    t = t.detach().float()
    min_t = torch.full_like(t, float(cfg.t1_min))
    if cfg.t1_sampler == "local_uniform":
        upper = (t - float(cfg.t1_min)).clamp_min(0.0).clamp_max(float(cfg.t1_delta))
        return (t - torch.rand_like(t) * upper).clamp_min(float(cfg.t1_min))

    if time_sampler is None:
        upper = (t - float(cfg.t1_min)).clamp_min(0.0).clamp_max(float(cfg.t1_delta))
        return (t - torch.rand_like(t) * upper).clamp_min(float(cfg.t1_min))

    out = time_sampler.sample(t.shape[0], device=device).float()
    for _ in range(cfg.t1_max_resample_attempts):
        bad = out >= t
        if not bool(bad.any()):
            break
        out[bad] = time_sampler.sample(int(bad.sum().item()), device=device).float()
    fallback_upper = (t - float(cfg.t1_min)).clamp_min(0.0)
    fallback = min_t + torch.rand_like(t) * fallback_upper
    out = torch.where(out < t, out, fallback)
    return out.clamp_min(float(cfg.t1_min))


def build_clean_loop_bank_input(
    *,
    cfg: CleanLoopConfig,
    process,
    xt: torch.Tensor,
    t: torch.Tensor,
    model_out: torch.Tensor,
    x0_hat: torch.Tensor,
    t1: torch.Tensor,
) -> torch.Tensor:
    if cfg.bank_input_mode == "step":
        return process.step(xt, t, t1, model_out, aux={})
    endpoint = torch.randn_like(x0_hat)
    alpha = process.schedule.alpha(t1)
    sigma = process.schedule.sigma(t1)
    a = process._reshape_coeff(alpha, x0_hat)
    s = process._reshape_coeff(sigma, x0_hat)
    return a * x0_hat + s * endpoint
