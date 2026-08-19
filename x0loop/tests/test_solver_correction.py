import torch

from x0loop.models.denoiser import Denoiser
from x0loop.models.solver_correction import SolverCorrectionConfig, SolverIndexCorrection


class _Net(torch.nn.Module):
    class Config:
        out_channels = 3

    cfg = Config()

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, t, cond=None):
        del t, cond
        return self.scale * x


def _config():
    return {"enabled": True, "solver_steps": 20, "start_index": 16, "hidden_channels": 8}


def test_solver_correction_is_zero_initialized_and_gated_to_last_four_intervals():
    module = SolverIndexCorrection(
        SolverCorrectionConfig(enabled=True, solver_steps=20, start_index=16, hidden_channels=8),
        channels=3,
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.8, 0.2])
    base = torch.randn_like(x)

    assert torch.equal(module(x, t, base), torch.zeros_like(x))
    with torch.no_grad():
        module.output.bias.fill_(1.0)
    result = module(x, t, base)
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    assert torch.equal(result[1], torch.ones_like(result[1]))
    assert module.solver_index(torch.tensor([1.0, 0.2, 0.0])).tolist() == [0, 16, 19]


def test_denoiser_correction_zero_init_is_functionally_identical_to_base():
    net = _Net()
    model = Denoiser(net, process=object(), solver_correction=_config())
    x = torch.randn(3, 3, 8, 8)
    t = torch.tensor([0.1, 0.5, 0.9])

    assert torch.equal(model(x, t), net(x, t))


def test_correction_only_forward_blocks_base_parameter_gradient():
    net = _Net()
    model = Denoiser(net, process=object(), solver_correction=_config())
    assert model.solver_correction is not None
    with torch.no_grad():
        model.solver_correction.output.bias.fill_(0.25)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    t = torch.full((2,), 0.1)

    loss = model(x, t, correction_only=True).square().mean()
    loss.backward()

    assert net.scale.grad is None
    assert model.solver_correction.output.bias.grad is not None
    assert x.grad is not None


def test_enabling_solver_correction_does_not_advance_global_cpu_rng():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    Denoiser(_Net(), process=object(), solver_correction=_config())
    actual = torch.rand(4)

    assert torch.equal(actual, expected)
