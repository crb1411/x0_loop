from __future__ import annotations

import torch

from x0loop.aug.base import BaseAugment


class NoAug(BaseAugment):
    def sample_params(self, batch_size: int, device=None, rng=None):
        return None

    def apply(self, x: torch.Tensor, params):
        return x
