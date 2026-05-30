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
    target: torch.Tensor           # what the model should directly predict
    aux: dict[str, Any] = field(default_factory=dict)


class BaseProcess:
    """
    output_target controls what the model predicts:
      "eps" — noise (default, classic DDPM / flow)
      "x0"  — clean image
      "v"   — velocity: v = α·ε − σ·x₀  (standard v-parameterization)

    The three from_output methods always return the requested quantity
    regardless of output_target, enabling any target for the loss config.
    """

    def __init__(self, schedule: TimeSchedule, prior: str = "gaussian", output_target: str = "eps"):
        self.schedule = schedule
        self.prior = prior
        self.output_target = str(output_target).lower()
        assert self.output_target in {"eps", "x0", "v"}, \
            f"output_target must be eps | x0 | v, got {self.output_target!r}"

    # ------------------------------------------------------------------ #
    # Internal converters: dispatch on self.output_target                  #
    # ------------------------------------------------------------------ #

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
            # x0 = (x_t − σ·ε) / α
            return (xt - s * model_out) / a.clamp_min(1e-5)
        # v = α·ε − σ·x0  →  x0 = (α·x_t − σ·v) / (α²+σ²)
        norm2 = self._reshape_coeff((alpha ** 2 + sigma ** 2).clamp_min(1e-10), xt)
        return (a * xt - s * model_out) / norm2

    def _to_eps(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        a, s, alpha, sigma = self._coeff(xt, t)
        if self.output_target == "eps":
            return model_out
        if self.output_target == "x0":
            # ε = (x_t − α·x0) / σ
            return (xt - a * model_out) / s.clamp_min(1e-5)
        # v = α·ε − σ·x0  →  ε = (σ·x_t + α·v) / (α²+σ²)
        norm2 = self._reshape_coeff((alpha ** 2 + sigma ** 2).clamp_min(1e-10), xt)
        return (s * xt + a * model_out) / norm2

    def _to_v(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        if self.output_target == "v":
            return model_out
        a, s, _, _ = self._coeff(xt, t)
        # standard v = α·ε − σ·x0
        return a * self._to_eps(xt, t, model_out) - s * self._to_x0(xt, t, model_out)

    # ------------------------------------------------------------------ #
    # Public API: from_output / targets                                    #
    # ------------------------------------------------------------------ #

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_x0(xt, t, model_out)

    def eps_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_eps(xt, t, model_out)

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_v(xt, t, model_out)

    def eps_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.aux["eps"]

    def x0_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.x0

    def v_target(self, fb: ForwardBatch) -> torch.Tensor:
        # standard v = α·ε − σ·x0
        alpha = self.schedule.alpha(fb.t)
        sigma = self.schedule.sigma(fb.t)
        a = self._reshape_coeff(alpha, fb.x0)
        s = self._reshape_coeff(sigma, fb.x0)
        return a * fb.aux["eps"] - s * fb.x0

    def _make_target(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the direct prediction target matching self.output_target."""
        if self.output_target == "eps":
            return eps
        if self.output_target == "x0":
            return x0
        # v = α·ε − σ·x0
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        return a * eps - s * x0

    # ------------------------------------------------------------------ #
    # Subclass interface                                                   #
    # ------------------------------------------------------------------ #

    def prior_sample(self, shape, device, dtype) -> torch.Tensor:
        if self.prior != "gaussian":
            raise ValueError(f"Unsupported prior: {self.prior}")
        return torch.randn(shape, device=device, dtype=dtype)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        raise NotImplementedError

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Sampling loop                                                        #
    # ------------------------------------------------------------------ #

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
