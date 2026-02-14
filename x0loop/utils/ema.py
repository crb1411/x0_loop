from __future__ import annotations

import copy

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
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            shadow_p = self.shadow[name]
            if shadow_p.device != p.device or shadow_p.dtype != p.dtype:
                shadow_p = shadow_p.to(device=p.device, dtype=p.dtype)
                self.shadow[name] = shadow_p
            new_avg = self.decay * shadow_p + (1.0 - self.decay) * p.detach()
            self.shadow[name] = new_avg.clone()

    def store(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()

    def copy_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                shadow_p = self.shadow[name]
                if shadow_p.device != p.device or shadow_p.dtype != p.dtype:
                    shadow_p = shadow_p.to(device=p.device, dtype=p.dtype)
                    self.shadow[name] = shadow_p
                p.data.copy_(shadow_p.data)

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                backup_p = self.backup[name]
                if backup_p.device != p.device or backup_p.dtype != p.dtype:
                    backup_p = backup_p.to(device=p.device, dtype=p.dtype)
                    self.backup[name] = backup_p
                p.data.copy_(backup_p.data)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}
