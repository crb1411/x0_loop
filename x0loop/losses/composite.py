from __future__ import annotations

from x0loop.losses.base import BaseLoss


class CompositeLoss(BaseLoss):
    def __init__(self, losses: list[BaseLoss], weights: list[float]):
        if len(losses) != len(weights):
            raise ValueError("losses and weights length mismatch")
        self.losses = losses
        self.weights = weights

    def __call__(self, model_out, target, *, t=None, aux=None):
        loss = 0.0
        for fn, w in zip(self.losses, self.weights):
            loss = loss + float(w) * fn(model_out, target, t=t, aux=aux)
        return loss
