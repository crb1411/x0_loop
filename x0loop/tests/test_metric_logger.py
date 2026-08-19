import pytest
import torch

from x0loop.utils.logger import MetricLogger


def test_metric_logger_keeps_all_cpu_tensor_samples():
    meters = MetricLogger(window_size=3)
    for value in (1.0, 2.0, 6.0):
        meters.update(loss=torch.tensor(value))

    assert meters.get_log_dict()["loss"] == pytest.approx(3.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA deferred metrics")
def test_metric_logger_batches_cuda_tensor_materialization_until_flush():
    meters = MetricLogger(window_size=3)
    meters.update(loss=torch.tensor(1.0, device="cuda"))
    meters.update(loss=torch.tensor(2.0, device="cuda"))

    assert "loss" not in meters.meters
    meters.flush_pending()
    assert meters.get_log_dict()["loss"] == pytest.approx(1.5)
