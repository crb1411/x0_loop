from experiments.x0loop_v2.analyze_training_efficiency import estimate_compute


def _config(*, enabled: bool, mode: str = "bank_fix") -> dict:
    return {
        "train": {"batch_size": 256},
        "clean_loop": {
            "enabled": enabled,
            "mode": mode,
            "aux_batch_ratio": 0.125,
            "solver_steps": 20,
            "root_fraction": 0.25,
            "refresh_interval": 1,
        },
    }


def test_fresh_compute_estimate_only_counts_main_training():
    estimate = estimate_compute(_config(enabled=False))
    assert estimate.fresh_forward_equivalent_samples == 768
    assert estimate.aux_forward_equivalent_samples == 0
    assert estimate.teacher_forward_equivalent_samples == 0


def test_bank_compute_estimate_counts_cfg_teacher_and_aux_backward():
    estimate = estimate_compute(_config(enabled=True, mode="bank_fix"))
    assert estimate.fresh_forward_equivalent_samples == 768
    assert estimate.aux_forward_equivalent_samples == 192
    assert estimate.teacher_forward_equivalent_samples == 160
    assert estimate.method_forward_equivalent_samples == 1120


def test_online_compute_estimate_counts_uniform_full_grid_occupancy():
    estimate = estimate_compute(_config(enabled=True, mode="online"))
    assert estimate.teacher_forward_equivalent_samples == 2 * 32 * (21 - 1 / 20)


def test_parameter_gradient_control_counts_two_measurement_vjps():
    cfg = _config(enabled=True, mode="online")
    cfg["clean_loop"]["aux_gradient_space"] = "parameter"

    estimate = estimate_compute(cfg)

    assert estimate.aux_forward_equivalent_samples == 6 * 32 + 2 * 256 + 4 * 32


def test_terminal_gan_counts_full_prefix_and_trainable_final_cfg_step():
    cfg = _config(enabled=False)
    cfg["adversarial"] = {
        "enabled": True,
        "fake_space": "terminal_x0",
        "batch_ratio": 0.125,
        "terminal": {"steps": 20},
    }

    estimate = estimate_compute(cfg)

    assert estimate.fresh_forward_equivalent_samples == 768
    assert estimate.teacher_forward_equivalent_samples == 4 * 32 * 19
    assert estimate.aux_forward_equivalent_samples == 6 * 32
    assert estimate.extra_flops_per_step > 0
