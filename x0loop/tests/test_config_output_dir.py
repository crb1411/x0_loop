from x0loop.core.config import resolve_logging_output_dir


def test_automatic_training_output_dir():
    cfg = {
        "dataset": {"name": "cifar10"},
        "model": {},
        "process": {"name": "flow", "output_target": "x0", "sampler": "heun"},
        "loss": {"terms": [{"target": "v", "formula": "mse"}]},
        "logging": {},
        "sample": {"sampler": "heun"},
    }

    resolve_logging_output_dir(cfg, timestamp="20260602_120000")

    assert cfg["logging"]["out_dir"] == "runs/cifar10/flow/dit/x0target_vloss_heun/20260602_120000"


def test_explicit_output_dir_is_preserved():
    cfg = {"logging": {"out_dir": "runs/debug"}}

    resolve_logging_output_dir(cfg, timestamp="20260602_120000")

    assert cfg["logging"]["out_dir"] == "runs/debug"


def test_flow_ddim_alias_is_named_euler():
    cfg = {
        "dataset": {"name": "cifar10"},
        "model": {"name": "unet"},
        "process": {"name": "flow", "output_target": "x0", "sampler": "ddim"},
        "loss": {"terms": [{"target": "v"}]},
        "logging": {"out_dir_base": "runs/legacy-value-is-ignored"},
        "sample": {"sampler": "ddim"},
    }

    resolve_logging_output_dir(cfg, timestamp="20260602_120000")

    assert cfg["logging"]["out_dir"] == "runs/cifar10/flow/unet/x0target_vloss_euler/20260602_120000"
