import torch

from x0loop.losses.adversarial import discriminator_loss, generator_loss, r1_penalty, t_weight
from x0loop.models.discriminator import X0Discriminator, X0DiscriminatorConfig
from x0loop.training.context import ForwardBatch
from x0loop.training.engine import maybe_apply_adversarial


class _DummyProcess:
    def x0_from_output(self, xt, t, out, aux):
        return out


class _DummyProcessBatch:
    def __init__(self, x0, xt, t):
        self.x0 = x0
        self.xt = xt
        self.t = t


def test_x0_discriminator_output_shape():
    disc = X0Discriminator(X0DiscriminatorConfig(base_channels=8, num_classes=10))
    x = torch.randn(4, 3, 32, 32)
    t = torch.rand(4)
    y = torch.tensor([0, 1, 9, 10])

    out = disc(x, t, y)

    assert out.shape == (4,)


def test_piecewise_t_weight_accepts_nested_list_config():
    cfg = {
        "adversarial": {
            "t_weight": {
                "name": "piecewise",
                "bins": [[0.0, 0.5, 1.0], [0.5, 1.0, 0.25]],
            }
        }
    }
    t = torch.tensor([0.1, 0.5, 1.0])

    w = t_weight(t, cfg)

    assert torch.allclose(w, torch.tensor([1.0, 0.25, 0.25]))


def test_hinge_adversarial_losses_are_per_example():
    real = torch.tensor([2.0, 0.5])
    fake = torch.tensor([-1.0, 0.25])

    d_total, d_real, d_fake = discriminator_loss(real, fake, loss="hinge")
    g = generator_loss(fake, loss="hinge")

    assert d_total.shape == d_real.shape == d_fake.shape == g.shape == (2,)
    assert torch.all(d_total >= 0)


def test_r1_penalty_is_per_example():
    x = torch.randn(3, 3, 4, 4, requires_grad=True)
    logits = x.square().mean(dim=(1, 2, 3))

    penalty = r1_penalty(logits, x)

    assert penalty.shape == (3,)
    assert torch.all(penalty >= 0)


def test_adversarial_branch_casts_bf16_fake_to_discriminator_dtype():
    disc = X0Discriminator(X0DiscriminatorConfig(base_channels=4, num_classes=10, spectral_norm=False))
    opt = torch.optim.AdamW(disc.parameters(), lr=1e-4)
    x0 = torch.randn(2, 3, 32, 32, dtype=torch.bfloat16)
    out = torch.randn(2, 3, 32, 32, dtype=torch.bfloat16, requires_grad=True)
    t = torch.tensor([0.2, 0.6])
    cond = torch.tensor([1, 3])
    fwd = ForwardBatch(
        loss=torch.zeros((), dtype=torch.float32),
        loss_by_target={},
        batch_size=2,
        cond=cond,
        fb=_DummyProcessBatch(x0=x0, xt=x0, t=t),
        out=out,
    )
    cfg = {
        "adversarial": {
            "enabled": True,
            "weight": 0.01,
            "start_step": 0,
            "warmup_steps": 0,
            "update_every": 1,
            "d_steps": 1,
            "loss": "hinge",
            "fake_space": "x0_hat",
            "r1": {"gamma": 0.0, "interval": 16},
            "t_weight": {"name": "uniform"},
        }
    }

    maybe_apply_adversarial(cfg=cfg, fwd=fwd, process=_DummyProcess(), discriminator=disc, d_optimizer=opt, scaler=None, step=0)
    fwd.loss.backward()

    assert out.grad is not None
    assert fwd.extra_metrics is not None
    assert fwd.extra_tbin is not None
