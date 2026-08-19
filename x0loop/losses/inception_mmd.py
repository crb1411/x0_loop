from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x

from x0loop.core.image_normalization import image_to_display_minus_one_one


def model_images_to_fid_pixels(
    images: torch.Tensor,
    cfg: dict,
    *,
    straight_through_quantize: bool,
) -> torch.Tensor:
    """Convert model-space images to the exact 8-bit values used by FID.

    With straight-through quantization the forward value is still the integer
    pixel produced by the evaluator, while the backward derivative is that of
    the unclipped floating-point pixel. Values outside the display range retain
    the evaluator's clamp and therefore receive zero derivative.
    """

    display = image_to_display_minus_one_one(images, cfg=cfg)
    pixels = (display.clamp(-1.0, 1.0) + 1.0) * 127.5
    if straight_through_quantize:
        return pixels + (pixels.floor() - pixels).detach()
    return pixels.to(torch.uint8)


class DifferentiableFIDInception(nn.Module):
    """The torch-fidelity FID network with a floating-point input contract.

    torch-fidelity deliberately requires uint8 at its public boundary. This
    wrapper copies only its forward graph after that dtype assertion, preserving
    the official weights and TensorFlow-compatible resize exactly.
    """

    def __init__(self, extractor: FeatureExtractorInceptionV3):
        super().__init__()
        if list(extractor.features_list) != ["2048"]:
            raise ValueError("DifferentiableFIDInception requires features_list=['2048']")
        self.extractor = extractor
        self.extractor.requires_grad_(False)
        self.extractor.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.extractor.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(pixels):
            raise TypeError(f"Expected floating-point FID pixels, got {pixels.dtype}")
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW FID pixels, got {tuple(pixels.shape)}")

        net = self.extractor
        x = pixels.to(net.feature_extractor_internal_dtype)
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(net.INPUT_IMAGE_SIZE, net.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        x = (x - 128) / 128
        x = net.Conv2d_1a_3x3(x)
        x = net.Conv2d_2a_3x3(x)
        x = net.Conv2d_2b_3x3(x)
        x = net.MaxPool_1(x)
        x = net.Conv2d_3b_1x1(x)
        x = net.Conv2d_4a_3x3(x)
        x = net.MaxPool_2(x)
        x = net.Mixed_5b(x)
        x = net.Mixed_5c(x)
        x = net.Mixed_5d(x)
        x = net.Mixed_6a(x)
        x = net.Mixed_6b(x)
        x = net.Mixed_6c(x)
        x = net.Mixed_6d(x)
        x = net.Mixed_6e(x)
        x = net.Mixed_7a(x)
        x = net.Mixed_7b(x)
        x = net.Mixed_7c(x)
        return torch.flatten(net.AvgPool(x), 1).to(torch.float32)


def polynomial_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    degree: int = 3,
    unbiased: bool = True,
) -> torch.Tensor:
    """Polynomial-kernel MMD^2 used by KID.

    The default kernel is ``(x @ y / d + 1) ** 3``. The U-statistic form is
    unbiased and is the form used for readiness and generator gradients.
    """

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected NxD and MxD features, got {tuple(x.shape)} and {tuple(y.shape)}")
    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}")
    if unbiased and (x.shape[0] < 2 or y.shape[0] < 2):
        raise ValueError("unbiased polynomial MMD requires at least two samples per distribution")
    dim = float(x.shape[1])
    k_xx = (x @ x.transpose(0, 1) / dim + 1.0).pow(degree)
    k_yy = (y @ y.transpose(0, 1) / dim + 1.0).pow(degree)
    k_xy = (x @ y.transpose(0, 1) / dim + 1.0).pow(degree)
    if unbiased:
        xx = (k_xx.sum() - k_xx.diagonal().sum()) / float(x.shape[0] * (x.shape[0] - 1))
        yy = (k_yy.sum() - k_yy.diagonal().sum()) / float(y.shape[0] * (y.shape[0] - 1))
    else:
        xx = k_xx.mean()
        yy = k_yy.mean()
    return xx + yy - 2.0 * k_xy.mean()

