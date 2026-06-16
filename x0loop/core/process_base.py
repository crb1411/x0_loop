from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from x0loop.core.schedules import TimeSchedule
from x0loop.core.types import expand_to_batch_image


@dataclass
class ForwardBatch:
    x0: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    # The second endpoint of the path x_t = alpha(t) x0 + sigma(t) endpoint.
    # For standard flow/diffusion this is Gaussian noise; for a learnable-endpoint
    # process it is the learned terminal point z.
    endpoint: torch.Tensor

    @property
    def eps(self) -> torch.Tensor:
        # Backward-compatible alias for code/tests that still read `.eps`.
        return self.endpoint


class BaseProcess(nn.Module):
    """Shared x_t = alpha(t) x0 + sigma(t) eps process utilities.

    output_target controls what the model directly predicts:
      eps       — noise endpoint
      x0        — clean image endpoint
      v         — velocity v = eps - x0
      velocity  — backward-compatible alias for v
    """

    VALID_TARGETS = {"eps", "x0", "v"}

    def __init__(self, schedule: TimeSchedule, prior: str = "gaussian", output_target: str = "eps"):
        super().__init__()
        self.schedule = schedule
        self.prior = prior
        self.output_target = self.normalize_target(output_target)

    @classmethod
    def normalize_target(cls, target: str) -> str:
        target = str(target).lower()
        if target in {"u", "flow", "flow_velocity", "velocity"}:
            target = "v"
        if target not in cls.VALID_TARGETS:
            raise ValueError(f"target must be eps | x0 | v, got {target!r}")
        return target

    def _coeff(self, xt: torch.Tensor, t: torch.Tensor):
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, xt)
        s = self._reshape_coeff(sigma, xt)
        return a, s, alpha, sigma

    def _to_x0(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        a, s, alpha, sigma = self._coeff(xt, t)
        if self.output_target == "x0":
            return model_out
        if self.output_target == "eps":
            return (xt - s * model_out) / a.clamp_min(1e-5)
        if self.output_target == "v":
            return (xt - s * model_out) / (a + s).clamp_min(1e-5)
        raise AssertionError(f"Unexpected output_target={self.output_target!r}")

    def _to_eps(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        a, s, alpha, sigma = self._coeff(xt, t)
        if self.output_target == "eps":
            return model_out
        if self.output_target == "x0":
            return (xt - a * model_out) / s.clamp_min(1e-5)
        if self.output_target == "v":
            return (xt + a * model_out) / (a + s).clamp_min(1e-5)
        raise AssertionError(f"Unexpected output_target={self.output_target!r}")

    def _to_v(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        if self.output_target == "v":
            return model_out
        return self._to_eps(xt, t, model_out) - self._to_x0(xt, t, model_out)

    def _to_velocity(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        return self._to_v(xt, t, model_out)

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_x0(xt, t, model_out)

    def endpoint_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_eps(xt, t, model_out)

    def eps_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        # Backward-compatible alias for `endpoint_from_output`.
        return self.endpoint_from_output(xt, t, model_out, aux)

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_v(xt, t, model_out)

    def velocity_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_velocity(xt, t, model_out)

    def x0_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.x0

    def endpoint_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.endpoint

    def eps_target(self, fb: ForwardBatch) -> torch.Tensor:
        # Backward-compatible alias for `endpoint_target`.
        return self.endpoint_target(fb)

    def v_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.endpoint - fb.x0

    def velocity_target(self, fb: ForwardBatch) -> torch.Tensor:
        return self.v_target(fb)

    def mudata_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} does not define a mudata prediction head.")

    def mudata_target(self, fb: ForwardBatch) -> torch.Tensor:
        raise ValueError(f"{type(self).__name__} does not define a mudata target.")

    def direct_target(self, fb: ForwardBatch) -> torch.Tensor:
        if self.output_target == "eps":
            return self.endpoint_target(fb)
        if self.output_target == "x0":
            return self.x0_target(fb)
        return self.v_target(fb)

    def prior_sample(self, shape, device, dtype) -> torch.Tensor:
        if self.prior != "gaussian":
            raise ValueError(f"Unsupported prior: {self.prior}")
        return torch.randn(shape, device=device, dtype=dtype)

    @staticmethod
    def guidance_scale_at_t(guidance_scale: float, t: torch.Tensor, guidance_schedule=None) -> float:
        max_scale = float(guidance_scale)
        if guidance_schedule is None:
            return max_scale
        if isinstance(guidance_schedule, str):
            guidance_schedule = {"name": guidance_schedule}
        if not isinstance(guidance_schedule, dict):
            raise TypeError(f"guidance_schedule must be a string or dict, got {type(guidance_schedule).__name__}")

        name = str(guidance_schedule.get("name", "constant")).lower()
        if name in {"", "none", "constant"}:
            return max_scale

        max_scale = float(guidance_schedule.get("max_scale", max_scale))
        min_scale = float(guidance_schedule.get("min_scale", 1.0))
        t_value = float(t.flatten()[0].detach().cpu())
        t_value = max(0.0, min(1.0, t_value))
        if name in {"power", "power_decay", "t_power"}:
            power = float(guidance_schedule.get("power", 1.0))
            weight = t_value ** power
        elif name in {"linear", "linear_decay"}:
            weight = t_value
        elif name in {"cosine", "cosine_decay"}:
            import math

            weight = 0.5 - 0.5 * math.cos(math.pi * t_value)
        else:
            raise ValueError(f"Unknown guidance_schedule.name={name!r}. Use constant|power_decay|linear_decay|cosine_decay.")
        return min_scale + (max_scale - min_scale) * weight

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        raise NotImplementedError

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, model, steps: int, shape, device, dtype, rng=None, return_trace: bool = False,
               cond=None, null_cond=None, guidance_scale: float = 1.0, sampler: str | None = None,
               posterior_noise_scale: float | None = None, guidance_schedule=None) -> dict:
        x = self.prior_sample(shape=shape, device=device, dtype=dtype)
        trace = []
        last_x0_hat = None
        for t_scalar, s_scalar in self.schedule.iter_pairs(steps=steps, device=device):
            t = torch.full((shape[0],), float(t_scalar.item()), device=device, dtype=torch.float32)
            out = model(x, t, cond=cond)
            step_guidance_scale = self.guidance_scale_at_t(guidance_scale, t, guidance_schedule)
            if cond is not None and null_cond is not None and float(step_guidance_scale) != 1.0:
                out_uncond = model(x, t, cond=null_cond)
                out = out_uncond + float(step_guidance_scale) * (out - out_uncond)
            xt = x
            x0_hat = self.x0_from_output(x, t, out, aux={})
            step_aux: dict[str, Any] = {}
            if sampler is not None:
                step_aux["sampler"] = sampler
            if posterior_noise_scale is not None:
                step_aux["posterior_noise_scale"] = float(posterior_noise_scale)
            x = self.step(x, t, s_scalar, out, aux=step_aux)
            last_x0_hat = x0_hat
            if return_trace:
                trace.append({
                    "t": t_scalar.detach().cpu(),
                    "guidance_scale": float(step_guidance_scale),
                    "x": xt.detach().cpu(),
                    "x0_hat": x0_hat.detach().cpu(),
                })
        result = {"x": x, "x0_hat": last_x0_hat}
        if return_trace:
            result["trace"] = trace
        return result

    @staticmethod
    def _reshape_coeff(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return expand_to_batch_image(v, x)
