import torch
import ssm_mbrl.common.modules as mod

nn = torch.nn


class InitialStateModel(torch.jit.ScriptModule):

    def __init__(self,
                 latent_state_dim: int,
                 learn_var: bool,
                 init_var: float,
                 min_var: float):

        super(InitialStateModel, self).__init__()

        self.register_buffer(name="_initial_mean",
                             tensor=torch.zeros(size=(1, latent_state_dim)))

        self._initial_variance_model = mod.ConstantUncertaintyLayer(dim=latent_state_dim,
                                                                    init_var=init_var,
                                                                    min_var=min_var,
                                                                    learn=learn_var)

    @torch.jit.script_method
    def forward(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {"mean": self._initial_mean.repeat(batch_size, 1),
                "var": self._initial_variance_model(batch_size),
                "sample": self._initial_mean.repeat(batch_size, 1)}
