import pytest

from x0loop.training.engine import build_loop_config


def _config(max_steps):
    return {
        "train": {
            "epochs": 3,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "lr": 1.0e-4,
            "max_steps": max_steps,
        },
        "logging": {},
    }


def test_max_steps_caps_full_epoch_schedule():
    loop = build_loop_config(_config(2), loader=range(5), distributed_cfg={})

    assert loop.optimizer_steps_per_epoch == 3
    assert loop.total_steps == 2


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError, match="train.max_steps must be > 0"):
        build_loop_config(_config(0), loader=range(5), distributed_cfg={})
