from __future__ import annotations

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch


class FlowProcess(BaseProcess):
    """Linear interpolation to Gaussian prior with z/eps regression target."""

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        z = torch.randn_like(x0)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        xt = a * x0 + s * z
        return ForwardBatch(x0=x0, t=t, xt=xt, target=z, aux={"z": z, "alpha": alpha, "sigma": sigma})

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        sigma = self.schedule.sigma(t)
        alpha = self.schedule.alpha(t)
        s = self._reshape_coeff(sigma, xt)
        a = self._reshape_coeff(alpha, xt)
        return (xt - s * model_out) / a.clamp_min(1e-5)

    def eps_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return model_out

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self.eps_from_output(xt, t, model_out, aux) - self.x0_from_output(xt, t, model_out, aux)

    def step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        model_out: torch.Tensor,
        aux: dict,
        rng=None,
    ) -> torch.Tensor:
        x0_hat = self.x0_from_output(xt, t, model_out, aux)
        if s.ndim == 0:
            s = torch.full((xt.shape[0],), float(s.item()), device=xt.device)
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)
        a_s = self._reshape_coeff(alpha_s, xt)
        s_s = self._reshape_coeff(sigma_s, xt)
        return a_s * x0_hat + s_s * model_out
