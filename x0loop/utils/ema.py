from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        shadow_params: list[torch.Tensor] = []
        model_params: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            shadow_p = self.shadow[name]
            if shadow_p.device != p.device or shadow_p.dtype != p.dtype:
                shadow_p = shadow_p.to(device=p.device, dtype=p.dtype)
                self.shadow[name] = shadow_p
            shadow_params.append(shadow_p)
            model_params.append(p.detach())
        if shadow_params:
            # One foreach launch replaces one allocation, one arithmetic launch,
            # and one clone per parameter. This is algebraically the same EMA.
            torch._foreach_lerp_(shadow_params, model_params, 1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module):
        backup_params: list[torch.Tensor] = []
        model_params: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            backup_p = self.backup.get(name)
            if (
                backup_p is None
                or backup_p.shape != p.shape
                or backup_p.device != p.device
                or backup_p.dtype != p.dtype
            ):
                backup_p = torch.empty_like(p)
                self.backup[name] = backup_p
            backup_params.append(backup_p)
            model_params.append(p.detach())
        if backup_params:
            torch._foreach_copy_(backup_params, model_params)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        model_params: list[torch.Tensor] = []
        shadow_params: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if name in self.shadow:
                shadow_p = self.shadow[name]
                if shadow_p.device != p.device or shadow_p.dtype != p.dtype:
                    shadow_p = shadow_p.to(device=p.device, dtype=p.dtype)
                    self.shadow[name] = shadow_p
                model_params.append(p)
                shadow_params.append(shadow_p)
        if model_params:
            torch._foreach_copy_(model_params, shadow_params)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        model_params: list[torch.Tensor] = []
        backup_params: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if name in self.backup:
                backup_p = self.backup[name]
                if backup_p.device != p.device or backup_p.dtype != p.dtype:
                    backup_p = backup_p.to(device=p.device, dtype=p.dtype)
                    self.backup[name] = backup_p
                model_params.append(p)
                backup_params.append(backup_p)
        if model_params:
            torch._foreach_copy_(model_params, backup_params)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}
