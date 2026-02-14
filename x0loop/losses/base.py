from __future__ import annotations

import torch


class BaseLoss:
    def compute_per_example(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, model_out, target, *, t=None, aux=None) -> torch.Tensor:
        return self.compute_per_example(model_out, target, t=t, aux=aux).mean()
