from __future__ import annotations

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch


class DiffusionProcess(BaseProcess):
    """VP diffusion process with deterministic DDIM and posterior sampling."""

    def __init__(self, schedule, prior: str = "gaussian", output_target: str = "eps",
                 sampler: str = "ddim", posterior_noise_scale: float = 1.0):
        super().__init__(schedule=schedule, prior=prior, output_target=output_target)
        self.sampler = str(sampler).lower()
        self.posterior_noise_scale = float(posterior_noise_scale)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        # Apply the forward noising equation x_t = alpha_t * x0 + sigma_t * eps.
        eps = torch.randn_like(x0)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        xt = a * x0 + s * eps
        return ForwardBatch(x0=x0, t=t, xt=xt, eps=eps)

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        # Decode the configured model target into both endpoints once per step.
        x0_hat  = self.x0_from_output(xt, t, model_out, aux)
        eps_hat = self.eps_from_output(xt, t, model_out, aux)
        if s.ndim == 0:
            s = torch.full((xt.shape[0],), float(s.item()), device=xt.device)
        sampler = str(aux.get("sampler", self.sampler)).lower()
        if sampler == "posterior":
            noise_scale = float(aux.get("posterior_noise_scale", self.posterior_noise_scale))
            return self._posterior_step(xt=xt, t=t, s=s, x0_hat=x0_hat, noise_scale=noise_scale)
        if sampler != "ddim":
            raise ValueError(f"Unknown diffusion sampler: {sampler!r}. Use 'ddim' or 'posterior'.")
        # DDIM deterministically reconstructs x_s with the predicted endpoints.
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)
        a_s = self._reshape_coeff(alpha_s, xt)
        s_s = self._reshape_coeff(sigma_s, xt)
        return a_s * x0_hat + s_s * eps_hat

    def _posterior_step(self, *, xt, t, s, x0_hat, noise_scale):
        # Compute the VP reverse posterior q(x_s | x_t, x0_hat).
        alpha_t = self.schedule.alpha(t)
        sigma_t = self.schedule.sigma(t)
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)

        a_t = self._reshape_coeff(alpha_t, xt)
        a_s = self._reshape_coeff(alpha_s, xt)

        ratio  = (alpha_t / alpha_s).clamp(min=1e-6, max=1.0 - 1e-6)
        beta2  = (1.0 - ratio.square()).clamp_min(1e-12)   # = σ_t² − r²σ_s² under VP
        sigma_t2 = sigma_t.square().clamp_min(1e-12)
        sigma_s2 = sigma_s.square().clamp_min(1e-12)

        k     = (sigma_s2 * ratio / sigma_t2).clamp_min(0.0)
        k_img = self._reshape_coeff(k, xt)
        mean  = a_s * x0_hat + k_img * (xt - a_t * x0_hat)

        if noise_scale <= 0.0:
            return mean

        # Scale only the posterior noise; the reverse-process mean stays fixed.
        post_var = (sigma_s2 * beta2 / sigma_t2).clamp_min(0.0)
        std = self._reshape_coeff(post_var.sqrt(), xt)
        return mean + float(noise_scale) * std * torch.randn_like(xt)
