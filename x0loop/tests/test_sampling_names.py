import torch

from x0loop.training.sampling import build_sample_label_names, save_trace_large_images


def test_trace_filename_includes_actual_class_name(tmp_path):
    trace = [{"t": torch.tensor(1.0), "x0_hat": torch.zeros(2, 3, 4, 4)}]
    names = build_sample_label_names({"dataset": {"name": "cifar10"}})
    save_trace_large_images(trace, str(tmp_path), "step_00000001", labels=torch.tensor([2, 3]), label_names=names)
    assert (tmp_path / "step_00000001_sample_000_ybird_x0loop.png").is_file()
    assert (tmp_path / "step_00000001_sample_001_ycat_x0loop.png").is_file()


def test_trace_filename_supports_unconditional_samples(tmp_path):
    trace = [{"t": torch.tensor(1.0), "x0_hat": torch.zeros(1, 3, 4, 4)}]
    save_trace_large_images(trace, str(tmp_path), "step_00000001")
    assert (tmp_path / "step_00000001_sample_000_x0loop.png").is_file()
