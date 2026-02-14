from __future__ import annotations

import torch
import torch.nn.functional as F

from x0loop.losses.base import BaseLoss


def _flatten_per_example(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).mean(dim=1)


class MSELoss(BaseLoss):
    def compute_per_example(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        return _flatten_per_example((model_out - target).pow(2))


class L1Loss(BaseLoss):
    def compute_per_example(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        return _flatten_per_example((model_out - target).abs())


class HuberLoss(BaseLoss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute_per_example(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        per_pixel = F.huber_loss(model_out, target, delta=self.delta, reduction="none")
        return _flatten_per_example(per_pixel)
