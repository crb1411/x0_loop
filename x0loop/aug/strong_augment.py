from __future__ import annotations

import torch
import torch.nn.functional as F

from x0loop.aug.base import BaseAugment


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: [3, H, W], in [0, 1]
    if x.shape[0] < 3:
        return x
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(3, 1, 1)
    g = (x[:3] * w).sum(dim=0, keepdim=True)
    return g.repeat(3, 1, 1)


class strongAugment(BaseAugment):
    """
    Strong image-model style augmentation that is safe for diffusion `data_only` mode.
    Recipe: random resized crop + hflip + color jitter + random grayscale + random erasing.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        crop_min_scale: float = 0.75,
        crop_max_scale: float = 1.0,
        crop_min_ratio: float = 0.75,
        crop_max_ratio: float = 1.3333,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        grayscale_prob: float = 0.1,
        erasing_prob: float = 0.25,
        erase_min_scale: float = 0.02,
        erase_max_scale: float = 0.2,
    ):
        self.hflip_prob = hflip_prob
        self.crop_min_scale = crop_min_scale
        self.crop_max_scale = crop_max_scale
        self.crop_min_ratio = crop_min_ratio
        self.crop_max_ratio = crop_max_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.grayscale_prob = grayscale_prob
        self.erasing_prob = erasing_prob
        self.erase_min_scale = erase_min_scale
        self.erase_max_scale = erase_max_scale

    def sample_params(self, batch_size: int, device=None, rng=None):
        del rng
        dev = device if device is not None else "cpu"
        return {
            "flip": (torch.rand(batch_size, device=dev) < self.hflip_prob),
            "crop_scale": torch.rand(batch_size, device=dev) * (self.crop_max_scale - self.crop_min_scale) + self.crop_min_scale,
            "crop_ratio": torch.rand(batch_size, device=dev) * (self.crop_max_ratio - self.crop_min_ratio) + self.crop_min_ratio,
            "crop_u": torch.rand(batch_size, device=dev),
            "crop_v": torch.rand(batch_size, device=dev),
            "brightness": torch.rand(batch_size, device=dev) * 2 * self.brightness + (1.0 - self.brightness),
            "contrast": torch.rand(batch_size, device=dev) * 2 * self.contrast + (1.0 - self.contrast),
            "saturation": torch.rand(batch_size, device=dev) * 2 * self.saturation + (1.0 - self.saturation),
            "grayscale": (torch.rand(batch_size, device=dev) < self.grayscale_prob),
            "erase": (torch.rand(batch_size, device=dev) < self.erasing_prob),
            "erase_scale": torch.rand(batch_size, device=dev) * (self.erase_max_scale - self.erase_min_scale) + self.erase_min_scale,
            "erase_u": torch.rand(batch_size, device=dev),
            "erase_v": torch.rand(batch_size, device=dev),
        }

    def apply(self, x: torch.Tensor, params):
        if params is None:
            return x
        b, c, h, w = x.shape
        out = []
        for i in range(b):
            xi = x[i]

            # Random resized crop
            scale = float(params["crop_scale"][i].item())
            ratio = float(params["crop_ratio"][i].item())
            area = h * w * scale
            crop_h = int(round((area / ratio) ** 0.5))
            crop_w = int(round((area * ratio) ** 0.5))
            crop_h = max(1, min(h, crop_h))
            crop_w = max(1, min(w, crop_w))
            max_top = max(0, h - crop_h)
            max_left = max(0, w - crop_w)
            top = int(round(float(params["crop_u"][i].item()) * max_top))
            left = int(round(float(params["crop_v"][i].item()) * max_left))
            xi = xi[:, top : top + crop_h, left : left + crop_w]
            xi = F.interpolate(xi.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]

            # HFlip
            if bool(params["flip"][i].item()):
                xi = torch.flip(xi, dims=[2])

            # Photometric jitter operates in [0, 1]
            yi = (xi + 1.0) * 0.5
            yi = yi * float(params["brightness"][i].item())
            mean = yi.mean(dim=(1, 2), keepdim=True)
            yi = (yi - mean) * float(params["contrast"][i].item()) + mean
            gray = _rgb_to_gray(yi)
            yi = gray + (yi - gray) * float(params["saturation"][i].item())
            if bool(params["grayscale"][i].item()) and c >= 3:
                yi = _rgb_to_gray(yi)
            yi = yi.clamp(0.0, 1.0)
            xi = yi * 2.0 - 1.0

            # Random erasing
            if bool(params["erase"][i].item()):
                erase_area = h * w * float(params["erase_scale"][i].item())
                erase_h = max(1, min(h, int(round(erase_area**0.5))))
                erase_w = max(1, min(w, int(round(erase_area**0.5))))
                max_top = max(0, h - erase_h)
                max_left = max(0, w - erase_w)
                etop = int(round(float(params["erase_u"][i].item()) * max_top))
                eleft = int(round(float(params["erase_v"][i].item()) * max_left))
                xi[:, etop : etop + erase_h, eleft : eleft + erase_w] = 0.0

            out.append(xi)
        return torch.stack(out, dim=0)
