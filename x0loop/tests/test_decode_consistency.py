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
    _check_process(DiffusionProcess(sched))


def test_decode_consistency_flow():
    sched = TimeSchedule(mode="flow", num_steps=1000)
    _check_process(FlowProcess(sched))
