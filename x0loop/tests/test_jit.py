import torch

from x0loop.models.factory import build_model
from x0loop.models.jit import JiT, JiTConfig, JiT_models


def test_jit_shape_and_cfg_null_label():
    cfg = JiTConfig(image_size=32, patch_size=4, dim=128, depth=2, heads=4, bottleneck_dim=32,
                    in_context_len=4, in_context_start=1, num_classes=10)
    model = JiT(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, torch.rand(2), cond=torch.tensor([0, 10]))
    assert out.shape == x.shape


def test_jit_registry_and_shared_factory():
    assert "JiT-B/16" in JiT_models
    model, cfg = build_model({"name": "jit", "image_size": 32, "patch_size": 4, "dim": 128, "depth": 2,
                              "heads": 4, "bottleneck_dim": 32, "in_context_len": 0})
    assert isinstance(model, JiT)
    assert cfg.image_size == 32
