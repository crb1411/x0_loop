import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.processes.flow_process import FlowProcess
from x0loop.training.metrics import TimeBinAccumulator


class ConstantX0Model(torch.nn.Module):
    def __init__(self, x0: torch.Tensor):
        super().__init__()
        self.register_buffer("x0", x0)
        self.calls = 0

    def forward(self, x, t, cond=None):
        del x, t, cond
        self.calls += 1
        return self.x0


class RecordingX0Model(torch.nn.Module):
    def __init__(self, x0: torch.Tensor):
        super().__init__()
        self.register_buffer("x0", x0)
        self.seen_t: list[torch.Tensor] = []

    def forward(self, x, t, cond=None):
        del x, cond
        self.seen_t.append(t.detach().clone())
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
            return_trace=True,
        )
        assert torch.allclose(result["x"], target, atol=1e-5)
        assert model.calls == expected_calls
        assert "x" in result["trace"][0]
        assert result["trace"][0]["x"].shape == target.shape


def test_flow_sampling_time_condition_shift_changes_model_time_only():
    shape = (2, 3, 4, 4)
    target = torch.randn(shape)
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = FlowProcess(schedule=schedule, output_target="x0", sampler="euler")
    model = RecordingX0Model(target)

    result = process.sample(
        model=model,
        steps=2,
        shape=shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampler="euler",
        return_trace=True,
        time_condition_shift={"shift": 0.02, "min_t": 0.001, "max_t": 0.999},
    )

    assert torch.allclose(result["trace"][0]["t"], torch.tensor(1.0))
    assert torch.allclose(result["trace"][0]["t_model"], torch.tensor(0.999))
    assert torch.allclose(model.seen_t[0], torch.full((shape[0],), 0.999))


def test_standard_flow_tbin_summary_uses_eps_label():
    shape = (2, 3, 4, 4)
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = FlowProcess(schedule=schedule, output_target="x0")
    x0 = torch.randn(shape)
    t = torch.tensor([0.25, 0.75])
    fb = process.forward_sample(x0=x0, t=t)
    out = torch.randn(shape)
    loss_fn = CompositeLoss([AtomicLoss(target="x0", formula="mse")])

    acc = TimeBinAccumulator(num_bins=2, device=x0.device)
    acc.update(schedule=schedule, process=process, loss_fn=loss_fn, fb=fb, out=out)
    summary = acc.summary(is_distributed=False)

    assert "leps=" in summary
    assert "lz=" not in summary
