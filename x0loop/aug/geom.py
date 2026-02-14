from __future__ import annotations

import torch
import torch.nn.functional as F

from x0loop.aug.base import BaseAugment


class GeomAugment(BaseAugment):
    def __init__(
        self,
        hflip_prob: float = 0.5,
        max_translation: int = 2,
        crop_min_scale: float = 0.9,
        enable_crop_resize: bool = True,
        random_crop_position: bool = False,
    ):
        self.hflip_prob = hflip_prob
        self.max_translation = max_translation
        self.crop_min_scale = crop_min_scale
        self.enable_crop_resize = enable_crop_resize
        self.random_crop_position = random_crop_position

    def sample_params(self, batch_size: int, device=None, rng=None):
        dev = device if device is not None else "cpu"
        flips = torch.rand(batch_size, device=dev) < self.hflip_prob
        tx = torch.randint(-self.max_translation, self.max_translation + 1, (batch_size,), device=dev)
        ty = torch.randint(-self.max_translation, self.max_translation + 1, (batch_size,), device=dev)
        scale = torch.rand(batch_size, device=dev) * (1.0 - self.crop_min_scale) + self.crop_min_scale
        crop_u = torch.rand(batch_size, device=dev)
        crop_v = torch.rand(batch_size, device=dev)
        return {"flips": flips, "tx": tx, "ty": ty, "scale": scale, "crop_u": crop_u, "crop_v": crop_v}

    def apply(self, x: torch.Tensor, params):
        if params is None:
            return x
        b, _, h, w = x.shape
        out = x

        # hflip
        flip_mask = params["flips"].view(b, 1, 1, 1)
        flipped = torch.flip(out, dims=[3])
        out = torch.where(flip_mask, flipped, out)

        # integer translation with roll
        translated = []
        for i in range(b):
            yi = int(params["ty"][i].item())
            xi = int(params["tx"][i].item())
            translated.append(torch.roll(out[i], shifts=(yi, xi), dims=(1, 2)))
        out = torch.stack(translated, dim=0)

        if not self.enable_crop_resize:
            return out

        # center crop with random scale then resize back
        cropped = []
        for i in range(b):
            s = float(params["scale"][i].item())
            ch = max(1, int(round(h * s)))
            cw = max(1, int(round(w * s)))
            if self.random_crop_position:
                max_top = max(0, h - ch)
                max_left = max(0, w - cw)
                top = int(round(float(params["crop_u"][i].item()) * max_top))
                left = int(round(float(params["crop_v"][i].item()) * max_left))
            else:
                top = max(0, (h - ch) // 2)
                left = max(0, (w - cw) // 2)
            patch = out[i : i + 1, :, top : top + ch, left : left + cw]
            patch = F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)
            cropped.append(patch[0])
        return torch.stack(cropped, dim=0)
