import torch
import pytest

from x0loop.losses.distribution import build_distribution_matching_config, distribution_matching_weight
from x0loop.models.denoiser import Denoiser
from x0loop.training.context import ForwardBatch
from x0loop.training.engine import (
    _terminal_correction_suffix,
    _terminal_heun_prefix_before_suffix,
    maybe_apply_distribution_matching,
)
from x0loop.core.schedules import TimeSchedule


def test_distribution_matching_registered_defaults_and_warmup():
    cfg = {
        "distribution_matching": {"enabled": True, "start_step": 10, "warmup_steps": 4},
        "gen_eval": {"steps": 20, "sampler": "heun", "guidance_scale": 2.2},
    }
    result = build_distribution_matching_config(cfg)

    assert result.objective == "inception_kid"
    assert result.suffix_steps == 4
    assert distribution_matching_weight(result, 9) == 0.0
    assert distribution_matching_weight(result, 10) == pytest.approx(0.25)
    assert distribution_matching_weight(result, 13) == 1.0


def test_distribution_matching_rejects_non_heun_or_invalid_suffix():
    with pytest.raises(ValueError, match="must be heun"):
        build_distribution_matching_config({
            "distribution_matching": {"terminal": {"sampler": "euler"}}
        })
    with pytest.raises(ValueError, match="suffix_steps"):
        build_distribution_matching_config({
            "distribution_matching": {"terminal": {"steps": 4, "suffix_steps": 5}}
        })


def test_detached_prefix_and_correction_suffix_match_heun_kernel():
    class Process:
        schedule = TimeSchedule(mode="flow", num_steps=1000)

        def velocity_from_output(self, x, t, out, aux):
            del x, t, aux
            return out

        def x0_from_output(self, x, t, out, aux):
            del x, t, aux
            return out

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.4))

        def model_output(self, x, t, **kwargs):
            del kwargs
            return (self.scale * (1.0 + t)).view(-1, 1, 1, 1).expand_as(x)

    process = Process()
    model = Model()
    root = torch.randn(2, 1, 2, 2)
    prefix = _terminal_heun_prefix_before_suffix(
        denoiser=model,
        process=process,
        root=root,
        steps=4,
        suffix_steps=2,
        cond=None,
        null_cond=None,
        guidance_scale=1.0,
    )
    actual = _terminal_correction_suffix(
        denoiser=model,
        process=process,
        x=prefix,
        steps=4,
        suffix_steps=2,
        cond=None,
        null_cond=None,
        guidance_scale=1.0,
    )

    expected = root.clone()
    pairs = process.schedule.iter_pairs(4, device=root.device)
    for index, (t, s) in enumerate(pairs):
        velocity_t = model.scale.detach() * (1.0 + t)
        if index == len(pairs) - 1:
            expected = expected + (s - t) * velocity_t
        else:
            velocity_s = model.scale.detach() * (1.0 + s)
            expected = expected + (s - t) * 0.5 * (velocity_t + velocity_s)
    assert torch.allclose(actual, expected)
    actual.square().mean().backward()
    assert model.scale.grad is not None and model.scale.grad.abs() > 0


def test_distribution_loss_is_parameter_gradient_controlled():
    class Net(torch.nn.Module):
        class Config:
            out_channels = 3

        cfg = Config()

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, t, cond=None):
            del t, cond
            return self.scale * x

    class Features(torch.nn.Module):
        def forward(self, pixels):
            return pixels.mean(dim=(2, 3))

    denoiser = Denoiser(
        Net(),
        process=object(),
        solver_correction={
            "enabled": True,
            "solver_steps": 20,
            "start_index": 16,
            "hidden_channels": 8,
        },
    )
    correction = denoiser.solver_correction
    assert correction is not None
    fake = correction.output.bias.view(1, 3, 1, 1).expand(4, 3, 2, 2)
    real = torch.randn_like(fake)
    fresh = sum(parameter.square().sum() for parameter in denoiser.parameters())
    fwd = ForwardBatch(
        loss=fresh,
        loss_by_target={},
        batch_size=4,
        cond=None,
        fb=object(),
        out=torch.zeros_like(fake),
        dist_real=real,
        dist_fake=fake,
    )
    cfg = {
        "dataset": {"name": "cifar10", "normalization": "minus_one_one"},
        "distribution_matching": {
            "enabled": True,
            "start_step": 0,
            "warmup_steps": 0,
            "gradient_ratio": 0.1,
            "scale_max": 1000.0,
        },
    }

    maybe_apply_distribution_matching(
        cfg=cfg,
        fwd=fwd,
        denoiser=denoiser,
        feature_extractor=Features(),
        step=0,
    )
    fwd.loss.backward()

    assert fwd.extra_metrics is not None
    assert float(fwd.extra_metrics["dist/gradient_ratio_actual"]) == pytest.approx(0.1, rel=1e-5)
    assert correction.output.bias.grad is not None
