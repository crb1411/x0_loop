from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from x0loop.core.process_base import ForwardBatch


@dataclass
class DenoiserBatch:
    fb: ForwardBatch
    out: torch.Tensor
    loss_dict: dict[str, torch.Tensor]
    cond: torch.Tensor | None = None


class Denoiser(nn.Module):
    """Thin JiT-style denoiser wrapper around a backbone and an x0-loop process.

    The backbone only implements f_theta(x_t, t, cond).  This wrapper owns the
    generative semantics: time sampling, noising path, model direct target,
    target-space conversion, loss construction, CFG, and sampling.
    """

    def __init__(self, net: nn.Module, *, process, loss_fn=None, time_sampler=None):
        super().__init__()
        self.net = net
        self.process = process
        self.loss_fn = loss_fn
        self.time_sampler = time_sampler

    @property
    def output_target(self) -> str:
        return self.process.output_target

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        return self.net(xt, t, cond=cond)

    def make_forward_batch(self, x0: torch.Tensor, t: torch.Tensor | None = None) -> ForwardBatch:
        if t is None:
            if self.time_sampler is None:
                raise ValueError("Denoiser.make_forward_batch requires t or a configured time_sampler.")
            t = self.time_sampler.sample(x0.shape[0], device=x0.device)
        return self.process.forward_sample(x0=x0, t=t)

    def compute_loss(self, x0: torch.Tensor, *, t: torch.Tensor | None = None, cond=None) -> DenoiserBatch:
        if self.loss_fn is None:
            raise ValueError("Denoiser.compute_loss requires loss_fn.")
        fb = self.make_forward_batch(x0, t=t)
        out = self.forward(fb.xt, fb.t, cond=cond)
        loss_dict = self.loss_fn(self.process, fb, out)
        return DenoiserBatch(fb=fb, out=out, loss_dict=loss_dict, cond=cond)

    def x0_from_output(self, xt: torch.Tensor, t: torch.Tensor, out: torch.Tensor, aux: dict | None = None) -> torch.Tensor:
        return self.process.x0_from_output(xt, t, out, aux or {})

    def eps_from_output(self, xt: torch.Tensor, t: torch.Tensor, out: torch.Tensor, aux: dict | None = None) -> torch.Tensor:
        return self.process.eps_from_output(xt, t, out, aux or {})

    def v_from_output(self, xt: torch.Tensor, t: torch.Tensor, out: torch.Tensor, aux: dict | None = None) -> torch.Tensor:
        return self.process.v_from_output(xt, t, out, aux or {})

    @staticmethod
    def cfg_combine(cond_out: torch.Tensor, uncond_out: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        return uncond_out + float(guidance_scale) * (cond_out - uncond_out)

    def model_output(self, x: torch.Tensor, t: torch.Tensor, *, cond=None, null_cond=None, guidance_scale: float = 1.0) -> torch.Tensor:
        out = self.forward(x, t, cond=cond)
        if cond is not None and null_cond is not None and float(guidance_scale) != 1.0:
            out_uncond = self.forward(x, t, cond=null_cond)
            out = self.cfg_combine(out, out_uncond, guidance_scale)
        return out

    @torch.no_grad()
    def sample(
        self,
        *,
        steps: int,
        shape,
        device,
        dtype,
        return_trace: bool = False,
        cond=None,
        null_cond=None,
        guidance_scale: float = 1.0,
        sampler: str | None = None,
        posterior_noise_scale: float | None = None,
    ) -> dict[str, Any]:
        return self.process.sample(
            model=self,
            steps=steps,
            shape=shape,
            device=device,
            dtype=dtype,
            return_trace=return_trace,
            cond=cond,
            null_cond=null_cond,
            guidance_scale=guidance_scale,
            sampler=sampler,
            posterior_noise_scale=posterior_noise_scale,
        )
