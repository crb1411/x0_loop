import torch

from x0loop.training.metrics import TimeBinAccumulator


def test_time_bin_summary_includes_extra_values():
    acc = TimeBinAccumulator(num_bins=2, device=torch.device("cpu"))
    t = torch.tensor([0.25, 0.75])

    acc.update_extra(
        t=t,
        values={
            "gadv": torch.tensor([1.0, 3.0]),
            "dacc": torch.tensor([0.5, 1.0]),
        },
    )

    summary = acc.summary(is_distributed=False)

    assert "gadv=1" in summary
    assert "gadv=3" in summary
    assert "dacc=0.5" in summary
    assert "dacc=1" in summary
