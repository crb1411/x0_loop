from __future__ import annotations

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch


class DiffusionProcess(BaseProcess):
    """Predict eps -> decode x0 -> deterministic DDIM-style step."""

    def __init__(self, schedule, prior: str = "gaussian", sampler: str = "ddim", posterior_noise_scale: float = 1.0):
        super().__init__(schedule=schedule, prior=prior)
        self.sampler = str(sampler).lower()
        self.posterior_noise_scale = float(posterior_noise_scale)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        eps = torch.randn_like(x0)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)

        xt = a * x0 + s * eps
        return ForwardBatch(x0=x0, t=t, xt=xt, target=eps, aux={"eps": eps, "alpha": alpha, "sigma": sigma})

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        sigma = self.schedule.sigma(t)
        alpha = self.schedule.alpha(t)
        s = self._reshape_coeff(sigma, xt)
        a = self._reshape_coeff(alpha, xt)
        return (xt - s * model_out) / a.clamp_min(1e-8)

    def eps_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return model_out

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self.eps_from_output(xt, t, model_out, aux) - self.x0_from_output(xt, t, model_out, aux)

    def eps_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.aux.get("eps", fb.target)

    def x0_target(self, fb: ForwardBatch) -> torch.Tensor:
        return fb.x0

    def v_target(self, fb: ForwardBatch) -> torch.Tensor:
        return self.eps_target(fb) - self.x0_target(fb)

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
        sampler = str(aux.get("sampler", self.sampler)).lower()
        if sampler == "posterior":
            noise_scale = float(aux.get("posterior_noise_scale", self.posterior_noise_scale))
            return self._posterior_step(xt=xt, t=t, s=s, x0_hat=x0_hat, noise_scale=noise_scale)
        if sampler != "ddim":
            raise ValueError(f"Unknown diffusion sampler: {sampler}. Expected 'ddim' or 'posterior'.")
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)
        a_s = self._reshape_coeff(alpha_s, xt)
        s_s = self._reshape_coeff(sigma_s, xt)
        return a_s * x0_hat + s_s * model_out

    def _posterior_step(
        self,
        *,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        x0_hat: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        # q(x_s | x_t, x0_hat) under VP forward process:
        # x_t = alpha(t) * x0 + sigma(t) * eps.
        alpha_t = self.schedule.alpha(t)
        sigma_t = self.schedule.sigma(t)
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)

        a_t = self._reshape_coeff(alpha_t, xt)
        a_s = self._reshape_coeff(alpha_s, xt)

        ratio = (alpha_t / alpha_s).clamp(min=1e-6, max=1.0 - 1e-6)
        ratio2 = ratio.square()
        beta2 = (1.0 - ratio2).clamp_min(1e-12)
        sigma_t2 = sigma_t.square().clamp_min(1e-12)
        sigma_s2 = sigma_s.square().clamp_min(1e-12)

        k = (sigma_s2 * ratio / sigma_t2).clamp_min(0.0)
        k_img = self._reshape_coeff(k, xt)
        mean = a_s * x0_hat + k_img * (xt - a_t * x0_hat)

        if noise_scale <= 0.0:
            return mean

        post_var = (sigma_s2 * beta2 / sigma_t2).clamp_min(0.0)
        std = self._reshape_coeff(post_var.sqrt(), xt)
        return mean + float(noise_scale) * std * torch.randn_like(xt)
