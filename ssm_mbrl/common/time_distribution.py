import torch
from typing import Optional

nn = torch.nn
jit = torch.jit


class _AbstractJittedTD(jit.ScriptModule):

    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, seq_length = x.shape[:2]
        bs = batch_size * seq_length
        new_shape = [bs, x.shape[2]] if len(x.shape) == 3 else [bs, x.shape[2], x.shape[3], x.shape[4]]
        return x.reshape(new_shape), batch_size, seq_length

    @staticmethod
    def _unflatten(x: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        if len(x.shape) == 2:
            new_shape = [batch_size, seq_length, x.shape[1]]
        else:
            new_shape = [batch_size, seq_length, x.shape[1], x.shape[2], x.shape[3]]
        return x.reshape(new_shape)

    @staticmethod
    def _get_full(valid: torch.Tensor, mask: torch.Tensor, default_value: int) -> torch.Tensor:
        full = torch.ones(size=mask.size()[:2] + valid.size()[1:], device=valid.device, dtype=valid.dtype)
        full *= default_value
        full[mask] = valid
        return full


class Jitted11TD(_AbstractJittedTD):

    def __init__(self, module):
        super(Jitted11TD, self).__init__()
        self._module = module

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, batch_size, seq_length = self._flatten(x)
        y_flat = self._module(x_flat)
        return self._unflatten(y_flat, batch_size, seq_length)


class Jitted12TD(_AbstractJittedTD):

    def __init__(self,
                 module):

        super(Jitted12TD, self).__init__()
        self._copy_through = getattr(module, "td_copy_trough", None)
        self._module = module

    @jit.script_method
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x_flat, batch_size, seq_length = self._flatten(x)
        y1_flat, y2_flat = self._module(x_flat)
        y1 = y1_flat if 0 in self._copy_through else self._unflatten(y1_flat, batch_size, seq_length)
        y2 = y2_flat if 1 in self._copy_through else self._unflatten(y2_flat, batch_size, seq_length)
        return y1, y2


class Jitted13TD(_AbstractJittedTD):

    def __init__(self,
                 module):

        super(Jitted13TD, self).__init__()
        self._copy_through = getattr(module, "td_copy_trough", [-1])
        self._module = module

    @jit.script_method
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat, batch_size, seq_length = self._flatten(x)
        y1_flat, y2_flat, y3_flat = self._module(x_flat)
        y1 = y1_flat if 0 in self._copy_through else self._unflatten(y1_flat, batch_size, seq_length)
        y2 = y2_flat if 1 in self._copy_through else self._unflatten(y2_flat, batch_size, seq_length)
        y3 = y3_flat if 2 in self._copy_through else self._unflatten(y3_flat, batch_size, seq_length)
        return y1, y2, y3


class _AbstractJittedTDPotentiallyMasked(_AbstractJittedTD):

    def __init__(self,
                 base_module: nn.Module,
                 mask_if_possible: bool = True,
                 default_value: int = 0):
        super(_AbstractJittedTDPotentiallyMasked, self).__init__()
        self._base_module = base_module
        self._mask_if_possible = mask_if_possible
        self._default_value = default_value
        self._copy_through = getattr(self._base_module, "td_copy_trough", None)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        if self._mask_if_possible and mask is not None:
            return self._forward_masked(x=x, mask=mask.reshape([x.shape[0], x.shape[1]]))
        else:
            return self._forward_unmasked(x=x)


class Jitted11TDPotentiallyMasked(_AbstractJittedTDPotentiallyMasked):

    @jit.script_method
    def _forward_unmasked(self, x: torch.Tensor):
        x_flat, batch_size, seq_length = self._flatten(x)
        y_flat = self._base_module(x_flat)
        return self._unflatten(y_flat, batch_size, seq_length)

    @jit.script_method
    def _forward_masked(self, x: torch.Tensor, mask: torch.Tensor):
        valid_outputs = self._base_module(x[mask])
        return self._get_full(valid_outputs, mask, self._default_value)


class Jitted12TDPotentiallyMasked(_AbstractJittedTDPotentiallyMasked):

    @jit.script_method
    def _forward_unmasked(self, x: torch.Tensor):
        x_flat, batch_size, seq_length = self._flatten(x)
        y1_flat, y2_flat = self._base_module(x_flat)
        y1 = y1_flat if 0 in self._copy_through else self._unflatten(y1_flat, batch_size, seq_length)
        y2 = y2_flat if 1 in self._copy_through else self._unflatten(y2_flat, batch_size, seq_length)
        return y1, y2

    @jit.script_method
    def _forward_masked(self, x: torch.Tensor, mask: torch.Tensor):
        y1_flat, y2_flat = self._base_module(x[mask])
        y1 = y1_flat if 0 in self._copy_through else self._get_full(y1_flat, mask, self._default_value)
        y2 = y2_flat if 1 in self._copy_through else self._get_full(y2_flat, mask, self._default_value)
        return y1, y2