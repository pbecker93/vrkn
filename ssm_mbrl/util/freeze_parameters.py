from typing import Union, Iterable
import torch

nn = torch.nn


class FreezeParameters:
    def __init__(self, modules: Union[Iterable[nn.Module], nn.Module]):
        self._param_states = []
        if isinstance(modules, nn.Module):
            self._params = list(modules.parameters())
        else:
            self._params = []
            for module in modules:
                self._params += list(module.parameters())

        self._param_states = [p.requires_grad for p in self._params]

    def __enter__(self):
        for param in self._params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for param, state in zip(self._params, self._param_states):
            param.requires_grad = state
