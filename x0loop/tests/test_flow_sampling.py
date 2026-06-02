import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.processes.flow_process import FlowProcess


class ConstantX0Model(torch.nn.Module):
    def __init__(self, x0: torch.Tensor):
        super().__init__()
        self.register_buffer("x0", x0)
        self.calls = 0

    def forward(self, x, t, cond=None):
        del x, t, cond
        self.calls += 1
        return self.x0


def test_flow_euler_and_heun_reach_constant_x0():
    shape = (2, 3, 4, 4)
    target = torch.randn(shape)
    schedule = TimeSchedule(mode="flow", num_steps=1000)

    for sampler, expected_calls in (("euler", 3), ("heun", 5), ("ddim", 3)):
        process = FlowProcess(schedule=schedule, output_target="x0", sampler=sampler)
        model = ConstantX0Model(target)
        result = process.sample(
            model=model,
            steps=3,
            shape=shape,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sampler=sampler,
        )
        assert torch.allclose(result["x"], target, atol=1e-5)
        assert model.calls == expected_calls
