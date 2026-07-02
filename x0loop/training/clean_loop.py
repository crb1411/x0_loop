from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CleanLoopConfig:
    enabled: bool = False
    bank_size: int = 50000
    bank_prob: float = 0.25
    warmup_steps: int = 10000
    loss_bank_weight: float = 0.3
    time_constant: float = 0.5
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
    if bank_size <= 0:
        raise ValueError(f"clean_loop.bank_size must be > 0, got {bank_size}")
    if not (0.0 <= bank_prob <= 1.0):
        raise ValueError(f"clean_loop.bank_prob must be in [0, 1], got {bank_prob}")
    if warmup_steps < 0:
        raise ValueError(f"clean_loop.warmup_steps must be >= 0, got {warmup_steps}")
    if loss_bank_weight < 0.0:
        raise ValueError(f"clean_loop.loss_bank_weight must be >= 0, got {loss_bank_weight}")
    return CleanLoopConfig(
        enabled=True,
        bank_size=bank_size,
        bank_prob=bank_prob,
        warmup_steps=warmup_steps,
        loss_bank_weight=loss_bank_weight,
        time_constant=time_constant,
    )


class CleanLoopBank:
    """Local FIFO ring buffer for (x0_hat, cond, x0) triples."""

    def __init__(self, cfg: CleanLoopConfig):
        self.cfg = cfg
        self.x_in: torch.Tensor | None = None
        self.x0: torch.Tensor | None = None
        self.cond: torch.Tensor | None = None
        self.steps: torch.Tensor | None = None
        self.next_idx = 0
        self.count = 0

    def __len__(self) -> int:
        return self.count

    @property
    def capacity(self) -> int:
        return self.cfg.bank_size

    def _allocate(self, x_in: torch.Tensor, x0: torch.Tensor, cond: torch.Tensor | None) -> None:
        shape = (self.capacity, *x_in.shape[1:])
        self.x_in = torch.empty(shape, dtype=self.cfg.storage_dtype, device="cpu")
        self.x0 = torch.empty(shape, dtype=self.cfg.storage_dtype, device="cpu")
        self.steps = torch.empty((self.capacity,), dtype=torch.long, device="cpu")
        if cond is not None:
            self.cond = torch.empty((self.capacity, *cond.shape[1:]), dtype=cond.dtype, device="cpu")

    @torch.no_grad()
    def add(self, *, x_in: torch.Tensor, x0: torch.Tensor, cond: torch.Tensor | None, step: int) -> None:
        if x_in.shape[0] == 0:
            return
        if self.x_in is None or self.x0 is None or self.steps is None:
            self._allocate(x_in, x0, cond)
        assert self.x_in is not None and self.x0 is not None and self.steps is not None

        n = int(x_in.shape[0])
        x_in_cpu = x_in.detach().to(device="cpu", dtype=self.cfg.storage_dtype)
        x0_cpu = x0.detach().to(device="cpu", dtype=self.cfg.storage_dtype)
        cond_cpu = cond.detach().to(device="cpu") if cond is not None else None

        for start in range(0, n, self.capacity):
            chunk_n = min(n - start, self.capacity)
            end = self.next_idx + chunk_n
            src = slice(start, start + chunk_n)
            if end <= self.capacity:
                dst = slice(self.next_idx, end)
                self.x_in[dst].copy_(x_in_cpu[src])
                self.x0[dst].copy_(x0_cpu[src])
                self.steps[dst].fill_(int(step))
                if self.cond is not None and cond_cpu is not None:
                    self.cond[dst].copy_(cond_cpu[src])
            else:
                first = self.capacity - self.next_idx
                second = chunk_n - first
                self.add(x_in=x_in_cpu[src][:first], x0=x0_cpu[src][:first], cond=cond_cpu[src][:first] if cond_cpu is not None else None, step=step)
                self.add(x_in=x_in_cpu[src][first:first + second], x0=x0_cpu[src][first:first + second], cond=cond_cpu[src][first:first + second] if cond_cpu is not None else None, step=step)
                continue
            self.next_idx = end % self.capacity
            self.count = min(self.capacity, self.count + chunk_n)

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if n <= 0 or self.count <= 0 or self.x_in is None or self.x0 is None or self.steps is None:
            raise ValueError("CleanLoopBank.sample requires a non-empty bank and n > 0")
        n = min(int(n), self.count)
        idx = torch.randint(self.count, (n,), device="cpu")
        x_in = self.x_in[idx].to(device=device, dtype=dtype)
        x0 = self.x0[idx].to(device=device, dtype=dtype)
        cond = self.cond[idx].to(device=device) if self.cond is not None else None
        steps = self.steps[idx].to(device=device)
        return x_in, cond, x0, steps
