import pytest
import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.core.time_sampling import build_time_sampler
from x0loop.train import _apply_set_overrides


def test_flow_grid_mix_can_use_solver_times_only():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    sampler = build_time_sampler(
        {
            "time_sampler": {
                "name": "uniform_continuous",
                "grid_mix_prob": 1.0,
                "grid_steps": [2, 4],
            }
        },
        schedule,
    )

    t = sampler.sample(256, device=torch.device("cpu"))
    allowed = torch.tensor([0.25, 0.5, 0.75, 1.0])
    assert torch.isin(t, allowed).all()


def test_flow_grid_mix_accepts_string_grid_steps_from_cli_override():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    sampler = build_time_sampler(
        {
            "time_sampler": {
                "name": "uniform_continuous",
                "grid_mix_prob": 1.0,
                "grid_steps": "[2,4]",
            }
        },
        schedule,
    )

    t = sampler.sample(256, device=torch.device("cpu"))
    allowed = torch.tensor([0.25, 0.5, 0.75, 1.0])
    assert torch.isin(t, allowed).all()


def test_set_override_parses_list_literals():
    cfg = {}

    _apply_set_overrides(cfg, ["time_sampler.grid_steps=[50,20]"])

    assert cfg["time_sampler"]["grid_steps"] == [50, 20]


def test_flow_grid_mix_zero_prob_keeps_base_sampler():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    sampler = build_time_sampler(
        {
            "time_sampler": {
                "name": "uniform_discrete",
                "num_steps": 10,
                "min_step": 3,
                "max_step": 3,
                "grid_mix_prob": 0.0,
                "grid_steps": [2],
            }
        },
        schedule,
    )

    t = sampler.sample(16, device=torch.device("cpu"))
    assert torch.allclose(t, torch.full_like(t, 0.3))


def test_grid_mix_rejects_diffusion_schedule():
    schedule = TimeSchedule(mode="diffusion", num_steps=1000)
    with pytest.raises(ValueError, match="only supported for flow"):
        build_time_sampler(
            {
                "time_sampler": {
                    "name": "uniform_continuous",
                    "grid_mix_prob": 0.5,
                    "grid_steps": [50, 20],
                }
            },
            schedule,
        )


def test_grid_mix_rejects_invalid_probability():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    with pytest.raises(ValueError, match="grid_mix_prob"):
        build_time_sampler(
            {
                "time_sampler": {
                    "name": "uniform_continuous",
                    "grid_mix_prob": -0.1,
                    "grid_steps": [50, 20],
                }
            },
            schedule,
        )
