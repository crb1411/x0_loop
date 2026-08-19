import torch

from x0loop.utils.ema import EMA


def test_ema_update_is_in_place_and_matches_reference_formula():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    ema = EMA(model, decay=0.75)
    shadow_ptrs = {name: value.data_ptr() for name, value in ema.shadow.items()}
    before = {name: value.clone() for name, value in ema.shadow.items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)

    for name, parameter in model.named_parameters():
        expected = 0.75 * before[name] + 0.25 * parameter
        assert torch.allclose(ema.shadow[name], expected)
        assert ema.shadow[name].data_ptr() == shadow_ptrs[name]


def test_ema_store_reuses_backup_buffers_and_restore_recovers_parameters():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, decay=0.9)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    ema.store(model)
    backup_ptrs = {name: value.data_ptr() for name, value in ema.backup.items()}
    ema.copy_to(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)
    ema.restore(model)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])

    ema.store(model)
    assert {name: value.data_ptr() for name, value in ema.backup.items()} == backup_ptrs
