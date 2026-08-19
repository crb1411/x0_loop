import torch

from x0loop.losses.inception_mmd import model_images_to_fid_pixels, polynomial_mmd2


def test_model_images_to_fid_pixels_matches_evaluator_quantization_with_straight_through_gradient():
    cfg = {"dataset": {"name": "cifar10", "normalization": "minus_one_one"}}
    base = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], requires_grad=True)
    images = base.view(1, 1, 1, 7).expand(1, 3, 1, 7)

    hard = model_images_to_fid_pixels(images, cfg, straight_through_quantize=False)
    differentiable = model_images_to_fid_pixels(images, cfg, straight_through_quantize=True)

    assert hard.dtype == torch.uint8
    assert torch.equal(differentiable.detach().to(torch.uint8), hard)
    differentiable.sum().backward()
    assert base.grad is not None
    assert base.grad[0].item() == 0.0
    assert base.grad[-1].item() == 0.0
    assert torch.all(base.grad[1:-1] > 0)


def test_polynomial_mmd2_matches_manual_unbiased_estimator_and_backpropagates():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    y = torch.tensor([[0.5, 0.0], [0.0, 0.5], [-0.5, -0.5]])
    dim = x.shape[1]
    kxx = (x @ x.T / dim + 1.0).pow(3)
    kyy = (y @ y.T / dim + 1.0).pow(3)
    kxy = (x @ y.T / dim + 1.0).pow(3)
    expected = (
        (kxx.sum() - kxx.diagonal().sum()) / 6
        + (kyy.sum() - kyy.diagonal().sum()) / 6
        - 2 * kxy.mean()
    )

    actual = polynomial_mmd2(x, y)

    assert torch.allclose(actual, expected)
    actual.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_biased_polynomial_mmd_is_zero_for_identical_features():
    x = torch.randn(5, 7)
    assert torch.allclose(polynomial_mmd2(x, x, unbiased=False), torch.zeros(()), atol=1e-6)
