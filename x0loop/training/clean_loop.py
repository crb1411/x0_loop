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
    t_bank: float = 0.75
    bank_input_mode: str = "x0_hat_resample"
    t1_sampler: str = "local_uniform"
    t1_delta: float = 0.02
    t1_min: float = 1.0e-5
    t1_max_resample_attempts: int = 8
    storage_dtype: torch.dtype = torch.float16
    version: int = 1
    mode: str = "legacy"
    aux_batch_ratio: float = 0.25
    aux_gradient_ratio: float = 0.2
    aux_scale_max: float = 10.0
    solver_steps: int = 20
    sampler: str = "heun"
    guidance_scale: float = 2.2
    root_fraction: float = 0.25
    drop_fraction: float = 0.5


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
    version = int(raw.get("version", 1))
    mode = str(raw.get("mode", "legacy" if version == 1 else "bank_fix")).lower()
    aux_batch_ratio = float(raw.get("aux_batch_ratio", raw.get("bank_prob", 0.25)))
    aux_gradient_ratio = float(raw.get("aux_gradient_ratio", 0.2))
    aux_scale_max = float(raw.get("aux_scale_max", 10.0))
    solver_steps = int(raw.get("solver_steps", 20))
    sampler = str(raw.get("sampler", "heun")).lower()
    guidance_scale = float(raw.get("guidance_scale", 2.2))
    root_fraction = float(raw.get("root_fraction", 0.25))
    drop_fraction = float(raw.get("drop_fraction", 0.5))
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
    if version not in {1, 2}:
        raise ValueError(f"clean_loop.version must be 1 or 2, got {version}")
    if mode not in {"legacy", "drop", "bank_fix", "online"}:
        raise ValueError("clean_loop.mode must be legacy | drop | bank_fix | online")
    if not (0.0 < aux_batch_ratio <= 1.0):
        raise ValueError(f"clean_loop.aux_batch_ratio must be in (0,1], got {aux_batch_ratio}")
    if not (0.0 <= aux_gradient_ratio <= 1.0):
        raise ValueError(f"clean_loop.aux_gradient_ratio must be in [0,1], got {aux_gradient_ratio}")
    if aux_scale_max <= 0.0:
        raise ValueError(f"clean_loop.aux_scale_max must be > 0, got {aux_scale_max}")
    if solver_steps <= 0:
        raise ValueError(f"clean_loop.solver_steps must be > 0, got {solver_steps}")
    if sampler != "heun":
        raise ValueError("clean_loop v2 currently locks the training kernel to sampler=heun")
    if not (0.0 <= root_fraction <= 1.0):
        raise ValueError(f"clean_loop.root_fraction must be in [0,1], got {root_fraction}")
    if not (0.0 <= drop_fraction < 1.0):
        raise ValueError(f"clean_loop.drop_fraction must be in [0,1), got {drop_fraction}")
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
        version=version,
        mode=mode,
        aux_batch_ratio=aux_batch_ratio,
        aux_gradient_ratio=aux_gradient_ratio,
        aux_scale_max=aux_scale_max,
        solver_steps=solver_steps,
        sampler=sampler,
        guidance_scale=guidance_scale,
        root_fraction=root_fraction,
        drop_fraction=drop_fraction,
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
    # Preserve the process-specific endpoint distribution.  In particular,
    # LearnableEndpointFlowProcess uses (1-beta) * mu_data + beta * noise.
    endpoint = process.prior_sample(x0_hat.shape, x0_hat.device, x0_hat.dtype)
    alpha = process.schedule.alpha(t1)
    sigma = process.schedule.sigma(t1)
    a = process._reshape_coeff(alpha, x0_hat)
    s = process._reshape_coeff(sigma, x0_hat)
    return a * x0_hat + s * endpoint


@dataclass
class TrajectoryBatch:
    x: torch.Tensor
    target_v: torch.Tensor
    t: torch.Tensor
    cond: torch.Tensor | None
    solver_index: torch.Tensor
    depth: torch.Tensor
    root_noise_id: torch.Tensor
    producer_step: torch.Tensor


class TrajectoryBank:
    """CPU replay for inference-kernel states and EMA velocity targets.

    Unlike the legacy FIFO, every item identifies its exact Heun grid point and
    trajectory ancestry. Sampling is stratified over solver index so roots do
    not crowd out the sampler's low-noise states.
    """

    def __init__(self, cfg: CleanLoopConfig):
        self.cfg = cfg
        self._items: list[dict[str, torch.Tensor | None]] = []
        self._next_root_id = 0

    def __len__(self) -> int:
        return len(self._items)

    def new_root_ids(self, n: int, *, device: torch.device) -> torch.Tensor:
        start = self._next_root_id
        self._next_root_id += int(n)
        return torch.arange(start, start + int(n), device=device, dtype=torch.long)

    @torch.no_grad()
    def add(self, batch: TrajectoryBatch) -> None:
        for i in range(batch.x.shape[0]):
            item: dict[str, torch.Tensor | None] = {
                "x": batch.x[i].detach().to(device="cpu", dtype=self.cfg.storage_dtype),
                "target_v": batch.target_v[i].detach().to(device="cpu", dtype=self.cfg.storage_dtype),
                "t": batch.t[i].detach().float().cpu(),
                "cond": batch.cond[i].detach().cpu() if batch.cond is not None else None,
                "solver_index": batch.solver_index[i].detach().long().cpu(),
                "depth": batch.depth[i].detach().long().cpu(),
                "root_noise_id": batch.root_noise_id[i].detach().long().cpu(),
                "producer_step": batch.producer_step[i].detach().long().cpu(),
            }
            self._items.append(item)
        overflow = len(self._items) - self.cfg.bank_size
        if overflow > 0:
            del self._items[:overflow]

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> TrajectoryBatch:
        if n <= 0 or not self._items:
            raise ValueError("TrajectoryBank.sample requires a non-empty bank and n > 0")
        by_index: dict[int, list[int]] = {}
        for pos, item in enumerate(self._items):
            index = int(item["solver_index"].item())  # type: ignore[union-attr]
            by_index.setdefault(index, []).append(pos)
        levels = list(by_index)
        chosen = [by_index[levels[i % len(levels)]][int(torch.randint(len(by_index[levels[i % len(levels)]]), ()).item())] for i in range(min(int(n), len(self._items)))]
        items = [self._items[i] for i in chosen]

        def stack(name: str) -> torch.Tensor:
            return torch.stack([item[name] for item in items])  # type: ignore[list-item]

        cond_items = [item["cond"] for item in items]
        cond = None if cond_items[0] is None else torch.stack(cond_items).to(device=device)  # type: ignore[arg-type]
        return TrajectoryBatch(
            x=stack("x").to(device=device, dtype=dtype),
            target_v=stack("target_v").to(device=device, dtype=dtype),
            t=stack("t").to(device=device),
            cond=cond,
            solver_index=stack("solver_index").to(device=device),
            depth=stack("depth").to(device=device),
            root_noise_id=stack("root_noise_id").to(device=device),
            producer_step=stack("producer_step").to(device=device),
        )
