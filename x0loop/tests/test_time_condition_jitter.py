import torch
from torch import nn

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.models.denoiser import Denoiser
from x0loop.processes.flow_process import FlowProcess


class RecordingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_t = None

    def forward(self, x, t, cond=None):
        del cond
        self.seen_t = t.detach().clone()
        return torch.zeros_like(x)


def test_time_condition_jitter_offsets_model_time_only():
    net = RecordingNet()
    process = FlowProcess(TimeSchedule(mode="flow", num_steps=1000), output_target="x0")
    loss_fn = CompositeLoss([AtomicLoss(target="v", formula="mse")])
    denoiser = Denoiser(
        net,
        process=process,
        loss_fn=loss_fn,
        time_condition_jitter={
            "enabled": True,
            "mean": 0.1,
            "std": 0.0,
            "prob": 1.0,
            "min_t": 1e-5,
            "max_t": 0.99999,
        },
    )

    x0 = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.95])
    batch = denoiser.compute_loss(x0, t=t)

    assert torch.allclose(batch.fb.t, t)
    assert torch.allclose(net.seen_t, torch.tensor([0.3, 0.99999]))


def test_model_conditioning_ignore_time_uses_constant_model_time_only():
    net = RecordingNet()
    process = FlowProcess(TimeSchedule(mode="flow", num_steps=1000), output_target="x0")
    loss_fn = CompositeLoss([AtomicLoss(target="v", formula="mse")])
    denoiser = Denoiser(
        net,
        process=process,
        loss_fn=loss_fn,
        time_condition_jitter={
            "enabled": True,
            "mean": 0.1,
            "std": 0.0,
            "prob": 1.0,
            "min_t": 1e-5,
            "max_t": 0.99999,
        },
        model_conditioning={
            "ignore_time": True,
            "time_constant": 0.5,
        },
    )

    x0 = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.95])
    batch = denoiser.compute_loss(x0, t=t)

    assert torch.allclose(batch.fb.t, t)
    assert torch.allclose(net.seen_t, torch.full_like(t, 0.5))
