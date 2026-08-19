import types

import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.processes.flow_process import LearnableEndpointFlowProcess
from x0loop.training.clean_loop import (
    CleanLoopBank,
    CleanLoopConfig,
    TrajectoryBank,
    TrajectoryBatch,
    build_clean_loop_bank_input,
    build_clean_loop_config,
    sample_clean_loop_t1,
)


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


def test_clean_loop_bank_fifo_capacity_and_sample_shapes():
    bank = CleanLoopBank(CleanLoopConfig(enabled=True, bank_size=3))
    x = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).view(5, 3, 2, 2)
    y = torch.arange(5, dtype=torch.long)
    t = torch.linspace(0.1, 0.5, 5)
    bank.add(x_in=x, x0=x + 1, t=t, cond=y, step=7)

    assert len(bank) == 3
    x_in, cond, x0, bank_t, steps = bank.sample(2, device=torch.device("cpu"), dtype=torch.float32)

    assert x_in.shape == (2, 3, 2, 2)
    assert x0.shape == (2, 3, 2, 2)
    assert bank_t.shape == (2,)
    assert cond is not None and cond.shape == (2,)
    assert steps.shape == (2,)
    assert torch.all(steps == 7)


def test_clean_loop_local_t1_sampler_stays_below_t():
    cfg = CleanLoopConfig(enabled=True, t1_sampler="local_uniform", t1_delta=0.02, t1_min=1e-3)
    t = torch.tensor([0.01, 0.2, 0.9])
    t1 = sample_clean_loop_t1(cfg=cfg, t=t, time_sampler=None, device=torch.device("cpu"))

    assert torch.all(t1 <= t)
    assert torch.all(t1 >= cfg.t1_min)
    assert torch.all((t - t1) <= cfg.t1_delta + 1e-6)


def test_clean_loop_resample_uses_process_endpoint_distribution():
    process = LearnableEndpointFlowProcess(
        TimeSchedule(mode="flow", num_steps=1000),
        image_size=2,
        data_channels=3,
        beta=0.8,
        output_target="x0",
    )
    endpoint = torch.full((2, 3, 2, 2), 3.0)
    process.prior_sample = types.MethodType(lambda self, shape, device, dtype: endpoint.to(device=device, dtype=dtype), process)
    x0_hat = torch.full_like(endpoint, 2.0)
    t1 = torch.tensor([0.25, 0.75])

    actual = build_clean_loop_bank_input(
        cfg=CleanLoopConfig(enabled=True, bank_input_mode="x0_hat_resample"),
        process=process,
        xt=torch.zeros_like(endpoint),
        t=torch.ones(2),
        model_out=x0_hat,
        x0_hat=x0_hat,
        t1=t1,
    )

    expected = (1.0 - t1[:, None, None, None]) * x0_hat + t1[:, None, None, None] * endpoint
    assert torch.allclose(actual, expected)


def test_trajectory_bank_preserves_v2_metadata():
    cfg = CleanLoopConfig(enabled=True, version=2, mode="bank_fix", bank_size=8)
    bank = TrajectoryBank(cfg)
    batch = TrajectoryBatch(
        x=torch.randn(4, 3, 2, 2),
        target_v=torch.randn(4, 3, 2, 2),
        t=torch.tensor([1.0, 0.75, 0.5, 0.25]),
        cond=torch.arange(4),
        solver_index=torch.arange(4),
        depth=torch.arange(4),
        root_noise_id=torch.tensor([10, 10, 10, 10]),
        producer_step=torch.full((4,), 7),
    )
    bank.add(batch)

    sampled = bank.sample(4, device=torch.device("cpu"), dtype=torch.float32)

    assert set(sampled.solver_index.tolist()) == {0, 1, 2, 3}
    assert torch.equal(sampled.depth, sampled.solver_index)
    assert torch.all(sampled.root_noise_id == 10)
    assert torch.all(sampled.producer_step == 7)
