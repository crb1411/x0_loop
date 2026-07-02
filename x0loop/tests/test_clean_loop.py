import torch

from x0loop.training.clean_loop import CleanLoopBank, CleanLoopConfig, build_clean_loop_config


def test_clean_loop_config_accepts_requested_aliases():
    cfg = build_clean_loop_config({
        "model_conditioning": {"time_constant": 0.5},
        "clean_loop": {
            "enabled": True,
            "max_num": 4,
            "p": 0.25,
            "warmupstep": 10000,
            "loss_bank_weight": 0.3,
        },
    })

    assert cfg.enabled
    assert cfg.bank_size == 4
    assert cfg.bank_prob == 0.25
    assert cfg.warmup_steps == 10000
    assert cfg.loss_bank_weight == 0.3
    assert cfg.time_constant == 0.5


def test_clean_loop_bank_fifo_capacity_and_sample_shapes():
    bank = CleanLoopBank(CleanLoopConfig(enabled=True, bank_size=3))
    x = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).view(5, 3, 2, 2)
    y = torch.arange(5, dtype=torch.long)
    bank.add(x_in=x, x0=x + 1, cond=y, step=7)

    assert len(bank) == 3
    x_in, cond, x0, steps = bank.sample(2, device=torch.device("cpu"), dtype=torch.float32)

    assert x_in.shape == (2, 3, 2, 2)
    assert x0.shape == (2, 3, 2, 2)
    assert cond is not None and cond.shape == (2,)
    assert steps.shape == (2,)
    assert torch.all(steps == 7)
