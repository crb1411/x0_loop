import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.losses.atomic import AtomicLoss, CompositeLoss
from x0loop.models.denoiser import Denoiser
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


class IncrementX0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []

    def forward(self, x, t, cond=None):
        del cond
        self.inputs.append(x.detach().clone())
        self.times.append(t.detach().clone())
        return x + 1.0


class RecordingIncrementNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.times: list[torch.Tensor] = []

    def forward(self, x, t, cond=None):
        del cond
        self.times.append(t.detach().clone())
        return x + 1.0


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


def test_flow_clean_loop_sampler_refines_x0hat_inputs():
    shape = (2, 3, 4, 4)
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = FlowProcess(schedule=schedule, output_target="x0", sampler="clean_loop")
    model = IncrementX0Model()

    result = process.sample(
        model=model,
        steps=3,
        shape=shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampler="clean_loop",
        refine_time=0.4,
        return_trace=True,
    )

    assert len(model.inputs) == 3
    assert len(result["trace"]) == 3
    assert torch.allclose(model.inputs[1], result["trace"][0]["x0_hat"])
    assert torch.allclose(model.inputs[2], result["trace"][1]["x0_hat"])
    assert torch.allclose(result["x"], result["trace"][2]["x0_hat"])
    assert torch.allclose(model.times[0], torch.ones_like(model.times[0]))
    assert torch.allclose(model.times[1], torch.full_like(model.times[1], 0.4))
    assert torch.allclose(model.times[2], torch.full_like(model.times[2], 0.4))


def test_flow_clean_loop_separates_path_t_and_model_t():
    shape = (2, 3, 4, 4)
    schedule = TimeSchedule(mode="flow", num_steps=1000)
    process = FlowProcess(schedule=schedule, output_target="x0", sampler="clean_loop")
    net = RecordingIncrementNet()
    denoiser = Denoiser(
        net,
        process=process,
        loss_fn=CompositeLoss([AtomicLoss(target="x0", formula="mse")]),
        model_conditioning={"ignore_time": True, "time_constant": 0.5},
    )

    result = process.sample(
        model=denoiser,
        steps=2,
        shape=shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampler="clean_loop",
        refine_time=0.4,
        return_trace=True,
    )

    assert torch.allclose(result["trace"][0]["path_t"], torch.tensor(1.0))
    assert torch.allclose(result["trace"][1]["path_t"], torch.tensor(0.4))
    assert torch.allclose(result["trace"][0]["model_t"], torch.tensor(0.5))
    assert torch.allclose(result["trace"][1]["model_t"], torch.tensor(0.5))
    assert torch.allclose(net.times[0], torch.full((shape[0],), 0.5))
    assert torch.allclose(net.times[1], torch.full((shape[0],), 0.5))


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
