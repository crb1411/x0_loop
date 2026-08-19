import types

import pytest
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
from x0loop.training.engine import _teacher_heun_step_batched, _teacher_targets
from x0loop.utils.checkpoint import _adapt_ema_state_to_model, _load_model_state_with_fallback


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


def test_clean_loop_refresh_interval_must_be_positive():
    with pytest.raises(ValueError, match="refresh_interval"):
        build_clean_loop_config({
            "clean_loop": {
                "enabled": True,
                "version": 2,
                "refresh_interval": 0,
            }
        })


def test_clean_loop_accepts_native_x0_aux_target():
    cfg = build_clean_loop_config({
        "clean_loop": {
            "enabled": True,
            "version": 2,
            "aux_target": "x0",
        }
    })
    assert cfg.aux_target == "x0"


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
        target_x0=torch.randn(4, 3, 2, 2),
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


def test_trajectory_bank_tensor_ring_keeps_latest_items():
    cfg = CleanLoopConfig(enabled=True, version=2, mode="bank_fix", bank_size=3)
    bank = TrajectoryBank(cfg)
    batch = TrajectoryBatch(
        x=torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1),
        target_v=torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1),
        target_x0=-torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1),
        t=torch.linspace(1.0, 0.2, 5),
        cond=torch.arange(5),
        solver_index=torch.arange(5),
        depth=torch.arange(5),
        root_noise_id=torch.arange(5),
        producer_step=torch.arange(5),
    )
    bank.add(batch)

    sampled = bank.sample(3, device=torch.device("cpu"), dtype=torch.float32)

    assert len(bank) == 3
    assert set(sampled.root_noise_id.tolist()) == {2, 3, 4}
    assert set(sampled.x.flatten().tolist()) == {2.0, 3.0, 4.0}
    assert set(sampled.target_x0.flatten().tolist()) == {-2.0, -3.0, -4.0}


def test_trajectory_bank_remainder_is_not_biased_to_early_solver_levels():
    cfg = CleanLoopConfig(enabled=True, version=2, mode="bank_fix", bank_size=20)
    bank = TrajectoryBank(cfg)
    levels = torch.arange(20)
    bank.add(TrajectoryBatch(
        x=levels.float().reshape(20, 1, 1, 1),
        target_v=torch.zeros(20, 1, 1, 1),
        target_x0=torch.ones(20, 1, 1, 1),
        t=torch.linspace(1.0, 0.05, 20),
        cond=None,
        solver_index=levels,
        depth=levels,
        root_noise_id=levels,
        producer_step=torch.zeros(20, dtype=torch.long),
    ))

    torch.manual_seed(1234)
    sampled_levels = torch.cat([
        bank.sample(12, device=torch.device("cpu"), dtype=torch.float32).solver_index
        for _ in range(400)
    ])

    # The former sorted remainder always had mean 5.5. Uniform level sampling
    # has expectation 9.5; use a wide deterministic bound to avoid a flaky test.
    assert 9.0 < sampled_levels.float().mean().item() < 10.0


def test_batched_teacher_heun_supports_heterogeneous_grid_pairs():
    class Denoiser:
        def model_output(self, x, t, **kwargs):
            del kwargs
            return t.reshape(-1, 1, 1, 1).expand_as(x)

    class Process:
        def velocity_from_output(self, x, t, output, aux):
            del x, t, aux
            return output

    x = torch.zeros(2, 1, 1, 1)
    t = torch.tensor([1.0, 0.5])
    s = torch.tensor([0.75, 0.0])
    actual, velocity, velocity_s = _teacher_heun_step_batched(
        denoiser=Denoiser(),
        process=Process(),
        x=x,
        t=t,
        s=s,
        cond=None,
        null_cond=None,
        guidance_scale=1.0,
    )

    expected = ((s - t) * 0.5 * (t + s)).reshape_as(x)
    assert torch.allclose(actual, expected)
    assert torch.equal(velocity.flatten(), t)
    assert torch.equal(velocity_s.flatten(), s)


def test_teacher_targets_preserve_direct_native_x0_output():
    class Denoiser:
        def model_output(self, x, t, **kwargs):
            del t, kwargs
            return torch.full_like(x, 3.0)

    class Process:
        def velocity_from_output(self, x, t, output, aux):
            del x, t, aux
            return output * 2.0

        def x0_from_output(self, x, t, output, aux):
            del x, t, aux
            return output

    velocity, target_x0 = _teacher_targets(
        denoiser=Denoiser(),
        process=Process(),
        x=torch.zeros(2, 1, 1, 1),
        t=torch.tensor([1.0, 0.5]),
        cond=None,
        null_cond=None,
        guidance_scale=1.0,
    )

    assert torch.all(velocity == 6.0)
    assert torch.all(target_x0 == 3.0)


def test_checkpoint_and_ema_load_across_compile_wrapper_prefix():
    class CompiledLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Module()
            self.net._orig_mod = torch.nn.Linear(2, 2)

    model = CompiledLike()
    eager_weight = torch.full_like(model.net._orig_mod.weight, 3.0)
    eager_bias = torch.full_like(model.net._orig_mod.bias, 4.0)
    eager = {"net.weight": eager_weight, "net.bias": eager_bias}

    info = _load_model_state_with_fallback(model, eager)
    ema_state = _adapt_ema_state_to_model(model, {"decay": 0.9, "shadow": eager})

    assert info["prefix"] == "net.->net._orig_mod."
    assert torch.equal(model.net._orig_mod.weight, eager_weight)
    assert set(ema_state["shadow"]) == {
        "net._orig_mod.weight",
        "net._orig_mod.bias",
    }
