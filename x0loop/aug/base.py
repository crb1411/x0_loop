from __future__ import annotations

import torch


class BaseAugment:
    def sample_params(self, batch_size: int, device=None, rng=None):
        raise NotImplementedError

    def apply(self, x: torch.Tensor, params):
        raise NotImplementedError
