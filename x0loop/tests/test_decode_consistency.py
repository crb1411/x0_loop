import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.processes.diffusion_process import DiffusionProcess
from x0loop.processes.flow_process import FlowProcess


@torch.no_grad()
def _check_process(proc):
    b, c, h, w = 8, 3, 16, 16
    x0 = torch.randn(b, c, h, w)
    t = torch.rand(b).clamp(1e-3, 0.999)

    fb = proc.forward_sample(x0=x0, t=t)
    perfect_out = fb.target
    x0_hat = proc.x0_from_output(fb.xt, fb.t, perfect_out, fb.aux)

    err = (x0_hat - x0).abs().mean().item()
    assert err < 1e-5, f"decode consistency error too high: {err}"


def test_decode_consistency_diffusion():
    sched = TimeSchedule(mode="diffusion", num_steps=1000)
    for target in ("eps", "x0", "v"):
        _check_process(DiffusionProcess(sched, output_target=target))


def test_decode_consistency_flow():
    sched = TimeSchedule(mode="flow", num_steps=1000)
    for target in ("eps", "x0", "v"):
        _check_process(FlowProcess(sched, output_target=target))


def test_v_target_is_xt_minus_x0_over_t():
    for proc in (
        DiffusionProcess(TimeSchedule(mode="diffusion", num_steps=1000)),
        FlowProcess(TimeSchedule(mode="flow", num_steps=1000)),
    ):
        x0 = torch.randn(4, 3, 8, 8)
        t = torch.rand(4).clamp(0.05, 0.999)
        fb = proc.forward_sample(x0=x0, t=t)
        expected = (fb.xt - fb.x0) / t.view(-1, 1, 1, 1)
        assert torch.allclose(proc.v_target(fb), expected, atol=1e-5, rtol=1e-5)


def test_flow_v_matches_eps_minus_x0():
    proc = FlowProcess(TimeSchedule(mode="flow", num_steps=1000))
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.rand(4).clamp(0.05, 0.999)
    fb = proc.forward_sample(x0=x0, t=t)
    assert torch.allclose(proc.v_target(fb), fb.aux["eps"] - fb.x0, atol=1e-5, rtol=1e-5)
