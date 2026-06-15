from __future__ import annotations

import torch
import pytest

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.weighted import make_weight_fn


def _values(name: str, **kwargs) -> torch.Tensor:
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    t = torch.tensor([0.001, 0.25, 0.5, 0.75, 0.999])
    return make_weight_fn(name, schedule=schedule, **kwargs)(t, None)


def test_core_time_weight_shapes_are_mean_normalized():
    assert torch.allclose(_values("none"), torch.ones(5))
    assert torch.allclose(
        _values("triangular", power=1.0, floor=0.0),
        torch.tensor([0.004, 1.0, 2.0, 1.0, 0.004]),
        atol=5e-3,
    )
    assert torch.allclose(
        _values("skew_triangular", power=1.0, floor=0.0, skew=0.5),
        torch.tensor([0.002, 0.75, 2.0, 1.25, 0.006]),
        atol=5e-3,
    )
    assert torch.allclose(
        _values("p2", p2_k=1.0, p2_gamma=1.0),
        torch.tensor([0.0, 0.2, 1.0, 1.8, 2.0]),
        atol=5e-3,
    )
    assert torch.allclose(
        _values("min_snr", gamma=5.0),
        torch.tensor([0.0, 0.716, 1.288, 1.288, 1.288]),
        atol=5e-3,
    )


def test_legacy_time_weight_names_are_rejected():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    for name in ("x0", "eps", "v", "target", "snr", "inv_snr", "logsnr", "balance_weights"):
        with pytest.raises(ValueError, match="Unknown weight fn"):
            make_weight_fn(name, schedule=schedule)
