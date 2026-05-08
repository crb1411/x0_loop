import torch

from x0loop.core.schedules import TimeSchedule
from x0loop.models.dit import DiT, DiTConfig
from x0loop.processes.diffusion_process import DiffusionProcess
from x0loop.processes.flow_process import FlowProcess


def _run_shape_test(mode: str):
    b, c, h, w = 2, 3, 32, 32
    x0 = torch.randn(b, c, h, w)

    cfg = DiTConfig(image_size=32, in_channels=3, out_channels=3, patch_size=4, dim=128, depth=2, heads=4, mlp_ratio=2.0)
    model = DiT(cfg)

    schedule = TimeSchedule(mode=mode, num_steps=1000)
    process = DiffusionProcess(schedule) if mode == "diffusion" else FlowProcess(schedule)

    t = schedule.sample_t(b, device=x0.device)
    fb = process.forward_sample(x0=x0, t=t)
    out = model(fb.xt, fb.t)
    assert out.shape == fb.target.shape == x0.shape

    s = torch.zeros((), dtype=torch.float32)
    x_next = process.step(fb.xt, fb.t, s=s, model_out=out, aux=fb.aux)
    assert x_next.shape == x0.shape


def test_shapes_diffusion():
    _run_shape_test("diffusion")


def test_shapes_flow():
    _run_shape_test("flow")


def test_dit_cfg_null_label():
    cfg = DiTConfig(
        image_size=32,
        in_channels=3,
        out_channels=3,
        patch_size=4,
        dim=128,
        depth=2,
        heads=4,
        mlp_ratio=2.0,
        num_classes=10,
    )
    model = DiT(cfg)
    assert model.label_emb.num_embeddings == 11

    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    cond = torch.tensor([0, 10])
    out = model(x, t, cond=cond)
    assert out.shape == x.shape
