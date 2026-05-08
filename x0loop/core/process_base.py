from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.core.types import expand_to_batch_image


@dataclass
class ForwardBatch:
    x0: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    target: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


class BaseProcess:
    def __init__(self, schedule: TimeSchedule, prior: str = "gaussian"):
        self.schedule = schedule
        self.prior = prior

    def prior_sample(self, shape, device, dtype) -> torch.Tensor:
        if self.prior != "gaussian":
            raise ValueError(f"Unsupported prior: {self.prior}")
        return torch.randn(shape, device=device, dtype=dtype)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        raise NotImplementedError

    def x0_from_output(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        model_out: torch.Tensor,
        aux: dict,
    ) -> torch.Tensor:
        raise NotImplementedError

    def step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        model_out: torch.Tensor,
        aux: dict,
        rng=None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        model,
        steps: int,
        shape,
        device,
        dtype,
        rng=None,
        return_trace: bool = False,
        cond=None,
        null_cond=None,
        guidance_scale: float = 1.0,
        sampler: str | None = None,
        posterior_noise_scale: float | None = None,
    ) -> dict:
        x = self.prior_sample(shape=shape, device=device, dtype=dtype)
        trace = []
        last_x0_hat = None

        for t_scalar, s_scalar in self.schedule.iter_pairs(steps=steps, device=device):
            t = torch.full((shape[0],), float(t_scalar.item()), device=device, dtype=torch.float32)
            out = model(x, t, cond=cond)
            if cond is not None and null_cond is not None and float(guidance_scale) != 1.0:
                out_uncond = model(x, t, cond=null_cond)
                out = out_uncond + float(guidance_scale) * (out - out_uncond)
            x0_hat = self.x0_from_output(x, t, out, aux={})
            step_aux: dict[str, Any] = {}
            if sampler is not None:
                step_aux["sampler"] = sampler
            if posterior_noise_scale is not None:
                step_aux["posterior_noise_scale"] = float(posterior_noise_scale)
            x = self.step(x, t, s_scalar, out, aux=step_aux)
            last_x0_hat = x0_hat
            if return_trace:
                trace.append({"t": t_scalar.detach().cpu(), "x0_hat": x0_hat.detach().cpu()})

        result = {"x": x, "x0_hat": last_x0_hat}
        if return_trace:
            result["trace"] = trace
        return result

    @staticmethod
    def _reshape_coeff(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return expand_to_batch_image(v, x)
