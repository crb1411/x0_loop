from __future__ import annotations


class Registry:
    def __init__(self):
        self._map = {}

    def register(self, name: str):
        def _inner(obj):
            self._map[name] = obj
            return obj

        return _inner

    def get(self, name: str):
        if name not in self._map:
            raise KeyError(f"{name} not found in registry")
        return self._map[name]


PROCESS_REGISTRY = Registry()
LOSS_REGISTRY = Registry()
AUG_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
