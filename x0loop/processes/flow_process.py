from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
        return ForwardBatch(x0=x0, t=t, xt=xt, endpoint=eps)

    def step(self, xt, t, s, model_out, aux, rng=None) -> torch.Tensor:
        # Freeze the predicted endpoints and reconstruct the same path at time s.
        x0_hat  = self.x0_from_output(xt, t, model_out, aux)
        eps_hat = self.endpoint_from_output(xt, t, model_out, aux)
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
        eps_hat = self.endpoint_from_output(xt, t, model_out, aux={})
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
            xt = x
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
                trace.append({"t": t_scalar.detach().cpu(), "x": xt.detach().cpu(), "x0_hat": x0_hat.detach().cpu()})

        result: dict[str, Any] = {"x": x, "x0_hat": last_x0_hat}
        if return_trace:
            result["trace"] = trace
        return result


class LearnableEndpointFlowProcess(FlowProcess):
    """Flow process with a learnable terminal endpoint.

    Training path:
        z   = (1 - beta) * mu_data + beta * noise
        x_t = (1 - t) * x0 + t * z

    The path endpoint (``ForwardBatch.endpoint`` / ``endpoint_target(fb)``) here
    is the learned, lightly-noised terminal point z, not raw Gaussian noise. The
    legacy ``eps``-named aliases still resolve to this same z for compatibility
    with diagnostics and target conversions.
    """

    def __init__(
        self,
        schedule,
        *,
        image_size: int,
        data_channels: int = 3,
        beta: float = 0.5,
        prior: str = "gaussian",
        output_target: str = "x0",
        sampler: str = "euler",
        predict_mudata: bool = False,
        mudata_init_mean: float = 0.0,
        mudata_init_std: float = 1.0,
        detach_mudata_target: bool = True,
    ):
        super().__init__(schedule=schedule, prior=prior, output_target=output_target, sampler=sampler)
        if not (0.0 <= float(beta) <= 1.0):
            raise ValueError(f"learnable endpoint beta must be in [0, 1], got {beta}")
        self.image_size = int(image_size)
        self.data_channels = int(data_channels)
        self.beta = float(beta)
        self.predict_mudata = bool(predict_mudata)
        self.detach_mudata_target = bool(detach_mudata_target)
        init = torch.randn(1, self.data_channels, self.image_size, self.image_size) * float(mudata_init_std) + float(mudata_init_mean)
        self.mu_data = nn.Parameter(init)

    def _main_output(self, model_out: torch.Tensor) -> torch.Tensor:
        return model_out[:, : self.data_channels]

    def _mu_output(self, model_out: torch.Tensor) -> torch.Tensor:
        if not self.predict_mudata:
            raise ValueError("mudata output was requested but process.predict_mudata=false.")
        if model_out.shape[1] < self.data_channels * 2:
            raise ValueError(
                f"predict_mudata=true requires at least {self.data_channels * 2} output channels, "
                f"got {model_out.shape[1]}."
            )
        return model_out[:, self.data_channels : self.data_channels * 2]

    def _terminal_endpoint(self, shape, device, dtype) -> torch.Tensor:
        noise = torch.randn(shape, device=device, dtype=dtype)
        mu = self.mu_data.to(device=device, dtype=dtype)
        return (1.0 - self.beta) * mu.expand(shape[0], -1, -1, -1) + self.beta * noise

    def prior_sample(self, shape, device, dtype) -> torch.Tensor:
        if self.prior != "gaussian":
            raise ValueError(f"Unsupported prior: {self.prior}")
        if tuple(shape[1:]) != (self.data_channels, self.image_size, self.image_size):
            raise ValueError(
                "LearnableEndpointFlowProcess prior shape must match "
                f"[B,{self.data_channels},{self.image_size},{self.image_size}], got {tuple(shape)}."
            )
        return self._terminal_endpoint(shape, device, dtype)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor, rng=None) -> ForwardBatch:
        del rng
        z = self._terminal_endpoint(x0.shape, x0.device, x0.dtype)
        alpha = self.schedule.alpha(t)
        sigma = self.schedule.sigma(t)
        a = self._reshape_coeff(alpha, x0)
        s = self._reshape_coeff(sigma, x0)
        xt = a * x0 + s * z
        return ForwardBatch(x0=x0, t=t, xt=xt, endpoint=z)

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_x0(xt, t, self._main_output(model_out))

    def endpoint_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_eps(xt, t, self._main_output(model_out))

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._to_v(xt, t, self._main_output(model_out))

    def velocity_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        return self.v_from_output(xt, t, model_out, aux)

    def mudata_from_output(self, xt: torch.Tensor, t: torch.Tensor, model_out: torch.Tensor, aux: dict) -> torch.Tensor:
        del xt, t, aux
        return self._mu_output(model_out)

    def mudata_target(self, fb: ForwardBatch) -> torch.Tensor:
        mu = self.mu_data.to(device=fb.x0.device, dtype=fb.x0.dtype)
        target = mu.expand(fb.x0.shape[0], -1, -1, -1)
        return target.detach() if self.detach_mudata_target else target
