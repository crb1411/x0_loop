from __future__ import annotations

import torch

from x0loop.losses.base import BaseLoss


class RuleLoss(BaseLoss):
    """Placeholder for future feature/physics/equivariance constraints."""

    def compute_per_example(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        return torch.zeros(model_out.shape[0], device=model_out.device, dtype=model_out.dtype)
