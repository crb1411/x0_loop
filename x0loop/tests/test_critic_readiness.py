from __future__ import annotations

import torch

from experiments.x0loop_v2.diagnose_critic_readiness import binary_auroc, _balanced_labels
from experiments.x0loop_v2.analyze_gradient_alignment import gradient_pair_metrics
from experiments.x0loop_v2.diagnose_fake_score_readiness import readiness_decision


def test_binary_auroc_perfect_reversed_and_tied() -> None:
    assert binary_auroc(torch.tensor([2.0, 3.0]), torch.tensor([-2.0, -1.0])) == 1.0
    assert binary_auroc(torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 3.0])) == 0.0
    assert binary_auroc(torch.zeros(3), torch.zeros(4)) == 0.5


def test_balanced_labels_are_deterministic_and_nearly_equal() -> None:
    left = _balanced_labels(103, 10, seed=7)
    right = _balanced_labels(103, 10, seed=7)
    assert torch.equal(left, right)
    counts = torch.bincount(left, minlength=10)
    assert int(counts.max() - counts.min()) <= 1


def test_gradient_pair_metrics_reports_cosine_and_scaled_ratio() -> None:
    fresh = [("net.blocks.0.weight", torch.tensor([3.0, 4.0]))]
    aux = [("net.blocks.0.weight", torch.tensor([-4.0, 3.0]))]
    metrics = gradient_pair_metrics(fresh, aux, scale=0.2)
    assert abs(metrics["cosine"]) < 1e-7
    assert abs(metrics["scaled_aux_to_fresh_norm"] - 0.2) < 1e-7
    assert metrics["groups"]["blocks_00_03"]["numel"] == 2


def test_fake_score_readiness_requires_aggregate_and_every_bin() -> None:
    teacher = {
        "x0_mse": 1.0,
        "v_mse": 2.0,
        "bins": {"a": {"x0_mse": 1.0, "v_mse": 2.0}},
    }
    passing = {
        "x0_mse": 0.8,
        "v_mse": 1.7,
        "bins": {"a": {"x0_mse": 0.9, "v_mse": 2.1}},
    }
    result = readiness_decision(teacher, passing, improvement_threshold=0.1, max_bin_regression=0.1)
    assert result["pass"]
    regressed = {**passing, "bins": {"a": {"x0_mse": 1.2, "v_mse": 2.1}}}
    result = readiness_decision(teacher, regressed, improvement_threshold=0.1, max_bin_regression=0.1)
    assert not result["pass"]
