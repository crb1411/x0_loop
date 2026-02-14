from __future__ import annotations

from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor


def expand_to_batch_image(v: Tensor, x: Tensor) -> Tensor:
    """Reshape [B] or scalar tensors so they broadcast over [B,C,H,W]."""
    if v.ndim == 0:
        return v.view(1, 1, 1, 1)
    if v.ndim == 1:
        return v.view(-1, 1, 1, 1)
    return v
