from __future__ import annotations

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch


class FlowProcess(BaseProcess):

    def __init__(self, schedule, prior: str = "gaussian", output_target: str = "eps"):
        super().__init__(schedule=schedule, prior=prior, output_target=output_target)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        eps = torch.randn_like(x0)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        xt = a * x0 + s * eps
        target = self._make_target(x0, eps, t)
        return ForwardBatch(x0=x0, t=t, xt=xt, target=target, aux={"eps": eps, "alpha": alpha, "sigma": sigma})

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        x0_hat  = self.x0_from_output(xt, t, model_out, aux)
        eps_hat = self.eps_from_output(xt, t, model_out, aux)
        if s.ndim == 0:
            s = torch.full((xt.shape[0],), float(s.item()), device=xt.device)
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)
        a_s = self._reshape_coeff(alpha_s, xt)
        s_s = self._reshape_coeff(sigma_s, xt)
        return a_s * x0_hat + s_s * eps_hat
