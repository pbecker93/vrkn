import torch
import math

nn = torch.nn


class DiagGaussActivation(nn.Module):

    def __init__(self,
                 init_var: float,
                 min_var: float) -> None:
        super(DiagGaussActivation, self).__init__()
        self._shift = self._get_shift(init_var=init_var, min_var=min_var)
        self._min_var = min_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self._shift) + self._min_var

    @staticmethod
    def _get_shift(init_var: float,
                   min_var: float) -> float:
        return math.log(math.exp(init_var - min_var) - 1)


class ScaledShiftedSigmoidActivation(nn.Module):

    def __init__(self,
                 init_val: float,
                 min_val: float,
                 max_val: float,
                 steepness: float = 1.0) -> None:
        super(ScaledShiftedSigmoidActivation, self).__init__()
        shift_init = init_val - min_val
        self._scale = max_val - min_val
        self._shift = math.log(shift_init / (self._scale - shift_init))
        self._min_val = min_val
        self._steepness = steepness

    def forward(self, x):
        return self._scale * torch.sigmoid(x * self._steepness + self._shift) + self._min_val
