from typing import Union, Iterable
import torch

jit = torch.jit
nn = torch.nn
F = torch.nn.functional


def build_hidden_layers_with_mc_dropout(in_features: int,
                                        drop_prob: float,
                                        layer_sizes: list[int],
                                        hidden_act: str = "ReLU") -> tuple[nn.ModuleList, int]:
    layers = []
    n_in = in_features
    n_out = n_in
    for layer_size in layer_sizes:
        n_out = layer_size
        layers.append(nn.Linear(in_features=n_in, out_features=n_out))
        layers.append(getattr(nn, hidden_act)())
        layers.append(MCDropout(p=drop_prob))
        n_in = n_out
    return nn.ModuleList(layers), n_out


class BayesianModule(jit.ScriptModule):

    def __init__(self):

        super(BayesianModule, self).__init__()
        self.use_map = False


class MCDropout(BayesianModule):

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(MCDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input=input,
                         p=self.p,
                         training=not self.use_map,  # Do dropout whenever we are not in "MAP"-mode
                         inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class MAPEstimate:

    def __init__(self, modules: Union[Iterable[nn.Module], nn.Module]):
        if isinstance(modules, nn.Module):
            self._modules = list(modules.modules())
        else:
            self._modules = []
            for module in modules:
                self._modules += list(module.modules())

    def __enter__(self):
        for module in self._modules:
            if isinstance(module, BayesianModule):
                module.use_map = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module in self._modules:
            if isinstance(module, BayesianModule):
                module.use_map = False
