from __future__ import annotations

import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.processes.flow_process import FlowProcess


def test_atomic_loss_call_applies_coef_like_composite_loss():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = FlowProcess(schedule=schedule, output_target="x0")
    x0 = torch.zeros(2, 3, 4, 4)
    t = torch.tensor([0.25, 0.75])
    fb = process.forward_sample(x0=x0, t=t)
    out = torch.ones_like(x0)
    atom = AtomicLoss(target="x0", formula="mse", coef=0.25)

    atomic_loss = atom(process, fb, out)
    composite_loss = CompositeLoss([atom])(process, fb, out)["total"]

    assert torch.allclose(atomic_loss, torch.tensor(0.25))
    assert torch.allclose(atomic_loss, composite_loss)
