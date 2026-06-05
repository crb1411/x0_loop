import torch

from x0loop.losses.adversarial import discriminator_loss, generator_loss, r1_penalty, t_weight
from x0loop.models.discriminator import X0Discriminator, X0DiscriminatorConfig


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
