from x0loop.core.config import dump_launch_config, resolve_logging_output_dir


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


def test_dump_launch_config_preserves_original_yaml_bytes(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_bytes(
        b"train:\n"
        b"  # keep this comment\n"
        b"  lr: 1.0e-4\n"
        b"logging:\n"
        b"  out_dir: null\n"
    )
    out_dir = tmp_path / "run"

    copied = dump_launch_config({"_config_path": str(source)}, str(out_dir))

    assert copied == str(out_dir / "launch_config.yaml")
    assert (out_dir / "launch_config.yaml").read_bytes() == source.read_bytes()


def test_dump_launch_config_returns_none_without_source(tmp_path):
    assert dump_launch_config({}, str(tmp_path / "run")) is None
