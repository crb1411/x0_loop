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
    aux_target: str = "velocity"
    solver_steps: int = 20
    sampler: str = "heun"
    guidance_scale: float = 2.2
    root_fraction: float = 0.25
    refresh_interval: int = 1
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
    aux_target = str(raw.get("aux_target", "velocity")).lower()
    solver_steps = int(raw.get("solver_steps", 20))
    sampler = str(raw.get("sampler", "heun")).lower()
    guidance_scale = float(raw.get("guidance_scale", 2.2))
    root_fraction = float(raw.get("root_fraction", 0.25))
    refresh_interval = int(raw.get("refresh_interval", 1))
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
    if aux_target not in {"velocity", "x0"}:
        raise ValueError(f"clean_loop.aux_target must be velocity | x0, got {aux_target!r}")
    if solver_steps <= 0:
        raise ValueError(f"clean_loop.solver_steps must be > 0, got {solver_steps}")
    if sampler != "heun":
        raise ValueError("clean_loop v2 currently locks the training kernel to sampler=heun")
    if not (0.0 <= root_fraction <= 1.0):
        raise ValueError(f"clean_loop.root_fraction must be in [0,1], got {root_fraction}")
    if refresh_interval <= 0:
        raise ValueError(f"clean_loop.refresh_interval must be > 0, got {refresh_interval}")
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
        aux_target=aux_target,
        solver_steps=solver_steps,
        sampler=sampler,
        guidance_scale=guidance_scale,
        root_fraction=root_fraction,
        refresh_interval=refresh_interval,
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
    target_x0: torch.Tensor
    t: torch.Tensor
    cond: torch.Tensor | None
    solver_index: torch.Tensor
    depth: torch.Tensor
    root_noise_id: torch.Tensor
    producer_step: torch.Tensor


class TrajectoryBank:
    """Tensor-ring replay for inference states and EMA velocity/native-x0 targets.

    Unlike the legacy FIFO, every item identifies its exact Heun grid point and
    trajectory ancestry. Sampling is stratified over solver index so roots do
    not crowd out the sampler's low-noise states. Formal runs keep the ring on
    the training GPU; CPU remains supported for unit tests and constrained
    environments.
    """

    def __init__(self, cfg: CleanLoopConfig, *, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.x: torch.Tensor | None = None
        self.target_v: torch.Tensor | None = None
        self.target_x0: torch.Tensor | None = None
        self.t: torch.Tensor | None = None
        self.cond: torch.Tensor | None = None
        self.solver_index: torch.Tensor | None = None
        self.depth: torch.Tensor | None = None
        self.root_noise_id: torch.Tensor | None = None
        self.producer_step: torch.Tensor | None = None
        self.next_idx = 0
        self.count = 0
        self._next_root_id = 0

    def __len__(self) -> int:
        return self.count

    def new_root_ids(self, n: int, *, device: torch.device) -> torch.Tensor:
        start = self._next_root_id
        self._next_root_id += int(n)
        return torch.arange(start, start + int(n), device=device, dtype=torch.long)

    def _allocate(self, batch: TrajectoryBatch) -> None:
        capacity = self.cfg.bank_size
        shape = (capacity, *batch.x.shape[1:])
        self.x = torch.empty(shape, device=self.device, dtype=self.cfg.storage_dtype)
        self.target_v = torch.empty(shape, device=self.device, dtype=self.cfg.storage_dtype)
        self.target_x0 = torch.empty(shape, device=self.device, dtype=self.cfg.storage_dtype)
        self.t = torch.empty(capacity, device=self.device, dtype=torch.float32)
        self.solver_index = torch.empty(capacity, device=self.device, dtype=torch.long)
        self.depth = torch.empty(capacity, device=self.device, dtype=torch.long)
        self.root_noise_id = torch.empty(capacity, device=self.device, dtype=torch.long)
        self.producer_step = torch.empty(capacity, device=self.device, dtype=torch.long)
        if batch.cond is not None:
            self.cond = torch.empty(
                (capacity, *batch.cond.shape[1:]),
                device=self.device,
                dtype=batch.cond.dtype,
            )

    @torch.no_grad()
    def add(self, batch: TrajectoryBatch) -> None:
        if batch.x.shape[0] == 0:
            return
        if self.x is None:
            self._allocate(batch)
        assert self.x is not None and self.target_v is not None and self.target_x0 is not None and self.t is not None
        assert self.solver_index is not None and self.depth is not None
        assert self.root_noise_id is not None and self.producer_step is not None

        n = int(batch.x.shape[0])
        offset = max(0, n - self.cfg.bank_size)
        n = min(n, self.cfg.bank_size)
        indices = (torch.arange(n, device=self.device) + self.next_idx) % self.cfg.bank_size

        def source(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
            return value[offset:].detach().to(device=self.device, dtype=dtype)

        self.x[indices] = source(batch.x, dtype=self.cfg.storage_dtype)
        self.target_v[indices] = source(batch.target_v, dtype=self.cfg.storage_dtype)
        self.target_x0[indices] = source(batch.target_x0, dtype=self.cfg.storage_dtype)
        self.t[indices] = source(batch.t, dtype=torch.float32)
        self.solver_index[indices] = source(batch.solver_index, dtype=torch.long)
        self.depth[indices] = source(batch.depth, dtype=torch.long)
        self.root_noise_id[indices] = source(batch.root_noise_id, dtype=torch.long)
        self.producer_step[indices] = source(batch.producer_step, dtype=torch.long)
        if batch.cond is not None:
            if self.cond is None:
                raise ValueError("TrajectoryBank conditioning changed after allocation")
            self.cond[indices] = source(batch.cond, dtype=self.cond.dtype)
        elif self.cond is not None:
            raise ValueError("TrajectoryBank conditioning changed after allocation")

        self.next_idx = (self.next_idx + n) % self.cfg.bank_size
        self.count = min(self.cfg.bank_size, self.count + n)

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> TrajectoryBatch:
        if n <= 0 or self.count == 0:
            raise ValueError("TrajectoryBank.sample requires a non-empty bank and n > 0")
        assert self.x is not None and self.target_v is not None and self.target_x0 is not None and self.t is not None
        assert self.solver_index is not None and self.depth is not None
        assert self.root_noise_id is not None and self.producer_step is not None

        sample_n = min(int(n), self.count)
        positions = torch.arange(self.count, device=self.device)
        stored_levels = self.solver_index[positions]
        levels = torch.unique(stored_levels, sorted=True)
        # Randomize which levels receive the remainder when sample_n is not a
        # multiple of the number of occupied levels. Cycling sorted levels made
        # a 32-sample/20-level bank assign every extra item to levels 0..11,
        # permanently biasing replay toward the high-noise half of Heun.
        level_order = levels[torch.randperm(levels.numel(), device=self.device)]
        wanted_levels = level_order[torch.arange(sample_n, device=self.device) % levels.numel()]
        matches = wanted_levels[:, None] == stored_levels[None, :]
        scores = torch.rand((sample_n, self.count), device=self.device)
        scores.masked_fill_(~matches, -1.0)
        chosen = positions[scores.argmax(dim=1)]

        def select(value: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
            selected = value[chosen]
            return selected.to(device=device, dtype=output_dtype or selected.dtype)

        cond = None if self.cond is None else select(self.cond)
        return TrajectoryBatch(
            x=select(self.x, output_dtype=dtype),
            target_v=select(self.target_v, output_dtype=dtype),
            target_x0=select(self.target_x0, output_dtype=dtype),
            t=select(self.t),
            cond=cond,
            solver_index=select(self.solver_index),
            depth=select(self.depth),
            root_noise_id=select(self.root_noise_id),
            producer_step=select(self.producer_step),
        )
