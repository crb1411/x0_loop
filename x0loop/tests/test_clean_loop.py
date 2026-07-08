import torch

from x0loop.training.clean_loop import CleanLoopBank, CleanLoopConfig, build_clean_loop_config, sample_clean_loop_t1


def test_clean_loop_config_accepts_requested_aliases():
    cfg = build_clean_loop_config({
        "model_conditioning": {"time_constant": 0.5},
        "clean_loop": {
            "enabled": True,
            "max_num": 4,
            "p": 0.25,
            "warmupstep": 10000,
            "loss_bank_weight": 0.3,
            "t_bank": 0.8,
            "bank_input_mode": "step",
            "t1_sampler": "local_uniform",
            "t1_delta": 0.02,
            "max_bank_readd_age": 1,
            "bank_loss": {"target": "v", "formula": "mse", "use_weight": True},
        },
    })

    assert cfg.enabled
    assert cfg.bank_size == 4
    assert cfg.bank_prob == 0.25
    assert cfg.warmup_steps == 10000
    assert cfg.loss_bank_weight == 0.3
    assert cfg.time_constant == 0.5
    assert cfg.t_bank == 0.8
    assert cfg.bank_input_mode == "step"
    assert cfg.t1_sampler == "local_uniform"
    assert cfg.t1_delta == 0.02
    assert cfg.max_bank_readd_age == 1.0
    assert cfg.bank_loss_target == "v"
    assert cfg.bank_loss_formula == "mse"
    assert cfg.bank_loss_use_weight


def test_clean_loop_bank_fifo_capacity_and_sample_shapes():
    bank = CleanLoopBank(CleanLoopConfig(enabled=True, bank_size=3))
    x = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).view(5, 3, 2, 2)
    y = torch.arange(5, dtype=torch.long)
    t = torch.linspace(0.1, 0.5, 5)
    bank.add(x_in=x, x0=x + 1, t=t, cond=y, age=1, label=y + 10)

    assert len(bank) == 3
    x_in, cond, x0, bank_t, ages, label = bank.sample(2, device=torch.device("cpu"), dtype=torch.float32)

    assert x_in.shape == (2, 3, 2, 2)
    assert x0.shape == (2, 3, 2, 2)
    assert bank_t.shape == (2,)
    assert cond is not None and cond.shape == (2,)
    assert label is not None and label.shape == (2,)
    assert ages.shape == (2,)
    assert torch.all(ages == 1)


def test_clean_loop_local_t1_sampler_stays_below_t():
    cfg = CleanLoopConfig(enabled=True, t1_sampler="local_uniform", t1_delta=0.02, t1_min=1e-3)
    t = torch.tensor([0.01, 0.2, 0.9])
    t1 = sample_clean_loop_t1(cfg=cfg, t=t, time_sampler=None, device=torch.device("cpu"))

    assert torch.all(t1 <= t)
    assert torch.all(t1 >= cfg.t1_min)
    assert torch.all((t - t1) <= cfg.t1_delta + 1e-6)


def test_clean_loop_uniform_below_t_sampler_uses_full_interval():
    torch.manual_seed(0)
    cfg = CleanLoopConfig(enabled=True, t1_sampler="uniform_below_t", t1_delta=0.02, t1_min=1e-3)
    t = torch.full((4096,), 0.9)
    t1 = sample_clean_loop_t1(cfg=cfg, t=t, time_sampler=None, device=torch.device("cpu"))

    assert torch.all(t1 <= t)
    assert torch.all(t1 >= cfg.t1_min)
    assert torch.any((t - t1) > cfg.t1_delta)


def test_clean_loop_fixed_delta_t1_sampler_uses_exact_delta():
    cfg = CleanLoopConfig(enabled=True, t1_sampler="fixed_delta", t1_delta=0.2, t1_min=1e-3)
    t = torch.tensor([0.1, 0.5, 0.9])
    t1 = sample_clean_loop_t1(cfg=cfg, t=t, time_sampler=None, device=torch.device("cpu"))

    assert torch.allclose(t1, torch.tensor([0.001, 0.3, 0.7]))
