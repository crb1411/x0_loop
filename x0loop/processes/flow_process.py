from __future__ import annotations

from typing import Any

import torch

from x0loop.core.process_base import BaseProcess, ForwardBatch


class FlowProcess(BaseProcess):
    """Linear flow matching process.

    The flow schedule follows x_t = (1 - t) * x0 + t * eps. Sampling starts
    from the Gaussian endpoint at t=1 and integrates backward to data at t=0.
    """

    def __init__(self, schedule, prior: str = "gaussian", output_target: str = "eps", sampler: str = "euler"):
        super().__init__(schedule=schedule, prior=prior, output_target=output_target)
        self.sampler = self._normalize_sampler(sampler)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        # Draw a point on the straight path between a clean sample and noise.
        eps = torch.randn_like(x0)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        xt = a * x0 + s * eps
        return ForwardBatch(x0=x0, t=t, xt=xt, eps=eps)

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        # Freeze the predicted endpoints and reconstruct the same path at time s.
        x0_hat  = self.x0_from_output(xt, t, model_out, aux)
        eps_hat = self.eps_from_output(xt, t, model_out, aux)
        if s.ndim == 0:
            s = torch.full((xt.shape[0],), float(s.item()), device=xt.device)
        alpha_s = self.schedule.alpha(s)
        sigma_s = self.schedule.sigma(s)
        a_s = self._reshape_coeff(alpha_s, xt)
        s_s = self._reshape_coeff(sigma_s, xt)
        return a_s * x0_hat + s_s * eps_hat

    @staticmethod
    def _normalize_sampler(sampler: str | None) -> str:
        name = "euler" if sampler is None else str(sampler).lower()
        # Keep old flow configs valid: their DDIM setting means Euler here.
        if name in {"auto", "ddim"}:
            return "euler"
        if name not in {"euler", "heun"}:
            raise ValueError(f"Unknown flow sampler: {sampler!r}. Use 'euler' or 'heun'.")
        return name

    def _velocity(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
        # Along the linear path, dx_t / dt = eps - x0.
        x0_hat = self.x0_from_output(xt, t, model_out, aux={})
        eps_hat = self.eps_from_output(xt, t, model_out, aux={})
        return eps_hat - x0_hat

    @staticmethod
    def _model_output(model, x, t, cond, null_cond, guidance_scale):
        out = model(x, t, cond=cond)
        if cond is not None and null_cond is not None and float(guidance_scale) != 1.0:
            out_uncond = model(x, t, cond=null_cond)
            out = out_uncond + float(guidance_scale) * (out - out_uncond)
        return out

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
        del rng, posterior_noise_scale
        method = self._normalize_sampler(self.sampler if sampler is None else sampler)
        x = self.prior_sample(shape=shape, device=device, dtype=dtype)
        trace = []
        last_x0_hat = None
        pairs = self.schedule.iter_pairs(steps=steps, device=device)

        for index, (t_scalar, s_scalar) in enumerate(pairs):
            t = torch.full((shape[0],), float(t_scalar.item()), device=device, dtype=torch.float32)
            out = self._model_output(model, x, t, cond, null_cond, guidance_scale)
            x0_hat = self.x0_from_output(x, t, out, aux={})
            velocity = self._velocity(x, t, out)
            # iter_pairs moves from t=1 to t=0, so dt is negative.
            dt = s_scalar - t_scalar

            # Keep the final step Euler: evaluating an x0-predicting model at t=0
            # would require an unstable eps reconstruction.
            if method == "heun" and index < len(pairs) - 1:
                # Heun averages the velocity before and after an Euler predictor.
                x_euler = x + dt * velocity
                s = torch.full((shape[0],), float(s_scalar.item()), device=device, dtype=torch.float32)
                out_s = self._model_output(model, x_euler, s, cond, null_cond, guidance_scale)
                velocity_s = self._velocity(x_euler, s, out_s)
                x = x + dt * 0.5 * (velocity + velocity_s)
            else:
                x = x + dt * velocity

            last_x0_hat = x0_hat
            if return_trace:
                trace.append({"t": t_scalar.detach().cpu(), "x0_hat": x0_hat.detach().cpu()})

        result: dict[str, Any] = {"x": x, "x0_hat": last_x0_hat}
        if return_trace:
            result["trace"] = trace
        return result
