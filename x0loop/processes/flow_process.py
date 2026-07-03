from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from x0loop.core.process_base import BaseProcess, ForwardBatch


class _UniPCSolver:
    """UniPC (Unified Predictor-Corrector) for the flow ODE, data-prediction form.

    Faithful port of diffusers' UniPCMultistepScheduler bh2 update with
    predict_x0=True. One model eval per step: the eval at the current sample
    both corrects the previous predictor output (UniC) and drives the next
    predictor (UniP). Order ramps up at the start and down to 1 at the end.
    """

    def __init__(self, order: int = 2, variant: str = "bh2"):
        self.order = int(order)
        self.variant = variant
        self.m_list: list[torch.Tensor] = []   # past data predictions x0_hat (recent last)
        self.lam_list: list[float] = []         # past log-SNR lambdas
        self.sig_list: list[float] = []          # past sigmas
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        self.lower_order_nums = 0

    @staticmethod
    def _solve(R: list[list[float]], b: list[float]) -> list[float]:
        Rt = torch.tensor(R, dtype=torch.float64)
        bt = torch.tensor(b, dtype=torch.float64)
        return torch.linalg.solve(Rt, bt).tolist()

    def _Rb(self, rks: list[float], hh: float, order: int, B_h: float):
        R: list[list[float]] = []
        b: list[float] = []
        h_phi_1 = math.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0
        factorial_i = 1.0
        for i in range(1, order + 1):
            R.append([rk ** (i - 1) for rk in rks])
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        return R, b

    def _B_h(self, hh: float) -> float:
        return math.expm1(hh) if self.variant == "bh2" else hh

    def correct(self, *, m_t, x_t, alpha_t, sigma_t, sigma_s0, lam_t, order):
        m0 = self.m_list[-1]
        lam_s0 = self.lam_list[-1]
        h = lam_t - lam_s0
        hh = -h
        h_phi_1 = math.expm1(hh)
        B_h = self._B_h(hh)
        rks: list[float] = []
        D1s: list[torch.Tensor] = []
        for i in range(1, order):
            rk = (self.lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((self.m_list[-(i + 1)] - m0) / rk)
        rks.append(1.0)
        R, b = self._Rb(rks, hh, order, B_h)
        rhos = [0.5] if order == 1 else self._solve(R, b)
        x_t_ = (sigma_t / sigma_s0) * self.last_sample - alpha_t * h_phi_1 * m0
        corr = sum((rhos[i] * D1s[i] for i in range(len(D1s))), start=torch.zeros_like(x_t))
        return x_t_ - alpha_t * B_h * (corr + rhos[-1] * (m_t - m0))

    def predict(self, *, m0, x, alpha_t, sigma_t, sigma_s0, lam_t, lam_s0, order):
        h = lam_t - lam_s0
        hh = -h
        h_phi_1 = math.expm1(hh)
        B_h = self._B_h(hh)
        rks: list[float] = []
        D1s: list[torch.Tensor] = []
        for i in range(1, order):
            rk = (self.lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((self.m_list[-(i + 1)] - m0) / rk)
        rks.append(1.0)
        R, b = self._Rb(rks, hh, order, B_h)
        x_t_ = (sigma_t / sigma_s0) * x - alpha_t * h_phi_1 * m0
        pred = torch.zeros_like(x)
        if D1s:
            rhos = [0.5] if order == 2 else self._solve([row[:-1] for row in R[:-1]], b[:-1])
            pred = sum((rhos[i] * D1s[i] for i in range(len(D1s))), start=torch.zeros_like(x))
        return x_t_ - alpha_t * B_h * pred


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
        # DPM-Solver++(2M) and its common spellings.
        if name in {"dpmpp", "dpmpp_2m", "dpm++", "dpm++2m", "dpm_solver++", "dpmsolver++"}:
            return "dpmpp_2m"
        # UniPC: "unipc" (2nd order) and "unipc3" (3rd order).
        if name in {"unipc", "unipc2", "uni_pc"}:
            return "unipc"
        if name in {"unipc3", "unipc_3"}:
            return "unipc3"
        if name in {"clean_loop", "cleanloop", "refine", "x0_refine", "x0hat_refine"}:
            return "clean_loop"
        if name not in {"euler", "heun", "dpmpp_2m", "unipc", "unipc3", "clean_loop"}:
            raise ValueError(
                "Unknown flow sampler: "
                f"{sampler!r}. Use 'euler', 'heun', 'dpmpp_2m', 'unipc', 'unipc3' or 'clean_loop'."
            )
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

    @staticmethod
    def _model_time(model, path_t: torch.Tensor) -> torch.Tensor:
        model_time_condition = getattr(model, "model_time_condition", None)
        if callable(model_time_condition):
            return model_time_condition(path_t)
        return path_t

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
        guidance_schedule=None,
        sampler: str | None = None,
        posterior_noise_scale: float | None = None,
        refine_time: float = 0.5,
    ) -> dict:
        del rng, posterior_noise_scale
        method = self._normalize_sampler(self.sampler if sampler is None else sampler)
        if method == "clean_loop":
            return self._sample_clean_loop_refine(
                model=model,
                steps=steps,
                shape=shape,
                device=device,
                dtype=dtype,
                return_trace=return_trace,
                cond=cond,
                null_cond=null_cond,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                refine_time=refine_time,
            )
        x = self.prior_sample(shape=shape, device=device, dtype=dtype)
        trace = []
        last_x0_hat = None
        pairs = self.schedule.iter_pairs(steps=steps, device=device)

        prev_x0_hat = None   # dpmpp_2m carries the previous step's data prediction.
        prev_lambda = None
        unipc = _UniPCSolver(order=3 if method == "unipc3" else 2) if method in {"unipc", "unipc3"} else None
        total = len(pairs)
        for index, (t_scalar, s_scalar) in enumerate(pairs):
            t = torch.full((shape[0],), float(t_scalar.item()), device=device, dtype=torch.float32)
            t_model = self._model_time(model, t)
            step_guidance_scale = self.guidance_scale_at_t(guidance_scale, t, guidance_schedule)
            out = self._model_output(model, x, t_model, cond, null_cond, step_guidance_scale)
            xt = x
            x0_hat = self.x0_from_output(x, t, out, aux={})
            is_last = index == len(pairs) - 1

            if unipc is not None:
                x = self._unipc_step(unipc, x=x, x0_hat=x0_hat, t_scalar=t_scalar,
                                     s_scalar=s_scalar, index=index, total=total)
            elif method == "dpmpp_2m":
                # Training-free multistep solver: 1 model eval per step, 2nd-order.
                x, prev_x0_hat, prev_lambda = self._dpmpp_2m_step(
                    x=x, t_scalar=t_scalar, s_scalar=s_scalar, x0_hat=x0_hat,
                    prev_x0_hat=prev_x0_hat, prev_lambda=prev_lambda, is_last=is_last,
                )
            else:
                velocity = self._velocity(x, t, out)
                # iter_pairs moves from t=1 to t=0, so dt is negative.
                dt = s_scalar - t_scalar
                # Keep the final step Euler: evaluating an x0-predicting model at t=0
                # would require an unstable eps reconstruction.
                if method == "heun" and not is_last:
                    # Heun averages the velocity before and after an Euler predictor.
                    x_euler = x + dt * velocity
                    s = torch.full((shape[0],), float(s_scalar.item()), device=device, dtype=torch.float32)
                    s_model = self._model_time(model, s)
                    s_guidance_scale = self.guidance_scale_at_t(guidance_scale, s, guidance_schedule)
                    out_s = self._model_output(model, x_euler, s_model, cond, null_cond, s_guidance_scale)
                    velocity_s = self._velocity(x_euler, s, out_s)
                    x = x + dt * 0.5 * (velocity + velocity_s)
                else:
                    x = x + dt * velocity

            last_x0_hat = x0_hat
            if return_trace:
                trace.append({
                    "t": t_scalar.detach().cpu(),
                    "path_t": t_scalar.detach().cpu(),
                    "model_t": t_model[0].detach().cpu(),
                    "guidance_scale": float(step_guidance_scale),
                    "x": xt.detach().cpu(),
                    "x0_hat": x0_hat.detach().cpu(),
                })

        result: dict[str, Any] = {"x": x, "x0_hat": last_x0_hat}
        if return_trace:
            result["trace"] = trace
        return result

    def _sample_clean_loop_refine(
        self,
        *,
        model,
        steps: int,
        shape,
        device,
        dtype,
        return_trace: bool,
        cond=None,
        null_cond=None,
        guidance_scale: float = 1.0,
        guidance_schedule=None,
        refine_time: float = 0.5,
    ) -> dict:
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"clean_loop sampler requires steps > 0, got {steps}")
        refine_time = float(refine_time)
        if not (0.0 <= refine_time <= 1.0):
            raise ValueError(f"clean_loop refine_time must be in [0,1], got {refine_time}")

        x = self.prior_sample(shape=shape, device=device, dtype=dtype)
        trace = []
        for index in range(steps):
            t_value = 1.0 if index == 0 else refine_time
            path_t = torch.full((shape[0],), t_value, device=device, dtype=torch.float32)
            model_t = self._model_time(model, path_t)
            step_guidance_scale = self.guidance_scale_at_t(guidance_scale, path_t, guidance_schedule)
            out = self._model_output(model, x, model_t, cond, null_cond, step_guidance_scale)
            x0_hat = self.x0_from_output(x, path_t, out, aux={})
            if return_trace:
                trace.append({
                    "t": path_t[0].detach().cpu(),
                    "path_t": path_t[0].detach().cpu(),
                    "model_t": model_t[0].detach().cpu(),
                    "guidance_scale": float(step_guidance_scale),
                    "x": x.detach().cpu(),
                    "x0_hat": x0_hat.detach().cpu(),
                })
            x = x0_hat

        result: dict[str, Any] = {"x": x, "x0_hat": x}
        if return_trace:
            result["trace"] = trace
        return result

    def _dpmpp_2m_step(self, *, x, t_scalar, s_scalar, x0_hat, prev_x0_hat, prev_lambda, is_last):
        """One DPM-Solver++(2M) data-prediction step (training-free, 1 NFE/step).

        Works in log-SNR lambda = log(alpha/sigma). The first-order update is
        x_s = (sigma_s/sigma_t) x_t + (alpha_s - alpha_t sigma_s/sigma_t) x0_hat,
        which equals the existing DDIM reconstruction; the multistep variant adds
        a correction from the previous step's data prediction. The first and last
        steps fall back to first order (lower_order_final) for stability.
        """
        a_t = float(self.schedule.alpha(t_scalar.reshape(1)))
        s_t = float(self.schedule.sigma(t_scalar.reshape(1)))
        a_s = float(self.schedule.alpha(s_scalar.reshape(1)))
        s_s = float(self.schedule.sigma(s_scalar.reshape(1)))
        lambda_t = math.log(a_t) - math.log(s_t)
        lambda_s = math.log(a_s) - math.log(s_s)
        h = lambda_s - lambda_t
        ratio = s_s / s_t
        coef = -a_s * (math.exp(-h) - 1.0)   # multiplies the data prediction

        if prev_x0_hat is None or is_last:
            x_next = ratio * x + coef * x0_hat
        else:
            r0 = (lambda_t - prev_lambda) / h
            d1 = (x0_hat - prev_x0_hat) / r0
            x_next = ratio * x + coef * (x0_hat + 0.5 * d1)
        return x_next, x0_hat, lambda_t

    def _asl(self, t_scalar):
        """Return (alpha, sigma, lambda=log(alpha/sigma)) scalars at time t."""
        a = float(self.schedule.alpha(t_scalar.reshape(1)))
        s = float(self.schedule.sigma(t_scalar.reshape(1)))
        return a, s, math.log(a) - math.log(s)

    def _unipc_step(self, solver: _UniPCSolver, *, x, x0_hat, t_scalar, s_scalar, index, total):
        """Drive one UniPC step: correct the previous predictor output, then predict."""
        a_cur, s_cur, lam_cur = self._asl(t_scalar)
        # Corrector reuses the current model eval to refine the previous prediction.
        if solver.last_sample is not None:
            x = solver.correct(
                m_t=x0_hat, x_t=x, alpha_t=a_cur, sigma_t=s_cur,
                sigma_s0=solver.sig_list[-1], lam_t=lam_cur, order=solver.this_order,
            )
        solver.m_list.append(x0_hat)
        solver.lam_list.append(lam_cur)
        solver.sig_list.append(s_cur)

        # lower_order_final: shrink order near the end; ramp up at the start.
        this_order = min(solver.order, total - index)
        this_order = min(this_order, solver.lower_order_nums + 1)
        solver.this_order = this_order
        solver.last_sample = x

        a_nxt, s_nxt, lam_nxt = self._asl(s_scalar)
        x = solver.predict(
            m0=x0_hat, x=x, alpha_t=a_nxt, sigma_t=s_nxt, sigma_s0=s_cur,
            lam_t=lam_nxt, lam_s0=lam_cur, order=this_order,
        )
        if solver.lower_order_nums < solver.order:
            solver.lower_order_nums += 1
        return x


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
