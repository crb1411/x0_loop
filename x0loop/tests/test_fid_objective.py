import torch

from x0loop.eval_fid import _build_denoiser
from x0loop.training.generative_eval import _cfg, _fixed_eval_rng


def test_gen_eval_seed_is_explicit_and_defaults_to_train_seed():
    assert _cfg({"train": {"seed": 17}})["seed"] == 17
    assert _cfg({"train": {"seed": 17}, "gen_eval": {"seed": 23}})["seed"] == 23


def test_fixed_eval_rng_does_not_change_training_rng_stream():
    torch.manual_seed(7)
    expected_first = torch.randn(4)
    expected_second = torch.randn(4)

    torch.manual_seed(7)
    actual_first = torch.randn(4)
    with _fixed_eval_rng(torch.device("cpu"), seed=123, rank=0):
        _ = torch.randn(100)
    actual_second = torch.randn(4)

    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)


def test_standalone_fid_preserves_model_time_conditioning():
    denoiser = _build_denoiser(
        {"model_conditioning": {"ignore_time": True, "time_constant": 0.5}},
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    actual = denoiser.model_time_condition(torch.tensor([0.1, 0.9]))
    assert torch.equal(actual, torch.tensor([0.5, 0.5]))
