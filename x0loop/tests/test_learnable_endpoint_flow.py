from __future__ import annotations

import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import CompositeLoss, AtomicLoss
from x0loop.processes.flow_process import LearnableEndpointFlowProcess
from x0loop.training.metrics import TimeBinAccumulator


def test_learnable_endpoint_flow_samples_from_mu_noise_endpoint():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = LearnableEndpointFlowProcess(schedule, image_size=4, data_channels=3, beta=0.5, output_target="x0")
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.tensor([0.25, 0.75])

    fb = process.forward_sample(x0, t)
    out = torch.randn(2, 3, 4, 4)

    assert fb.xt.shape == x0.shape
    assert fb.eps.shape == x0.shape
    assert process.x0_from_output(fb.xt, fb.t, out, aux={}).shape == x0.shape
    assert process.prior_sample((2, 3, 4, 4), x0.device, x0.dtype).shape == x0.shape


def test_learnable_endpoint_flow_mudata_head_loss_and_tbin():
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = LearnableEndpointFlowProcess(
        schedule,
        image_size=4,
        data_channels=3,
        beta=0.5,
        output_target="x0",
        predict_mudata=True,
    )
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.tensor([0.25, 0.75])
    fb = process.forward_sample(x0, t)
    out = torch.randn(2, 6, 4, 4)

    loss_fn = CompositeLoss([AtomicLoss(target="x0", formula="mse"), AtomicLoss(target="mudata", formula="mse")])
    losses = loss_fn(process, fb, out)
    assert "mudata" in losses

    acc = TimeBinAccumulator(num_bins=2, device=x0.device)
    acc.update(schedule=schedule, process=process, loss_fn=loss_fn, fb=fb, out=out)
    assert "lmu=" in acc.summary(is_distributed=False)
