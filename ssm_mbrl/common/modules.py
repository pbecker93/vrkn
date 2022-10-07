from typing import Optional, Union
import torch
import ssm_mbrl.common.activation as va

nn = torch.nn


class SimpleGaussianParameterLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 distribution_dim: int,
                 min_std_or_var: float = 1e-6):
        super(SimpleGaussianParameterLayer, self).__init__()
        self._mean_model = nn.Linear(in_features=in_features,
                                     out_features=distribution_dim)
        self._std_or_var_model = nn.Linear(in_features=in_features,
                                           out_features=distribution_dim)
        self._min_std_or_var = min_std_or_var

    def forward(self, hidden_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mean_model(hidden_features), \
               nn.functional.softplus(self._std_or_var_model(hidden_features)) + self._min_std_or_var


class DiagonalGaussianParameterLayer(nn.Module):

    valid_output_normalizations = ["none", "post", "post_ln", "pre", "pre_ln"]

    def __init__(self,
                 in_features: int,
                 distribution_dim: int,
                 init_var: float = 1.0,
                 min_var: float = 1e-6,
                 max_var: float = 100,
                 sigmoid_activation: bool = False,
                 output_normalization: Optional[str] = None):

        super(DiagonalGaussianParameterLayer, self).__init__()

        output_normalization = "none" if output_normalization is None else output_normalization.lower()
        assert output_normalization in DiagonalGaussianParameterLayer.valid_output_normalizations
        elementwise_affine = "ln" in output_normalization

        if sigmoid_activation:
            std_activation = va.ScaledShiftedSigmoidActivation(init_val=init_var,
                                                               min_val=min_var,
                                                               max_val=max_var)
        else:
            std_activation = va.DiagGaussActivation(init_var=init_var,
                                                    min_var=min_var)

        if "pre" in output_normalization:
            self._mean_net = nn.Sequential(
                nn.LayerNorm(normalized_shape=in_features, elementwise_affine=elementwise_affine),
                nn.Linear(in_features=in_features, out_features=distribution_dim),
                nn.Tanh())
            self._std_net = nn.Sequential(
                nn.LayerNorm(normalized_shape=in_features, elementwise_affine=elementwise_affine),
                nn.Linear(in_features=in_features, out_features=distribution_dim),
                std_activation)

        elif "post" in output_normalization:
            self._mean_net = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=distribution_dim),
                nn.LayerNorm(normalized_shape=distribution_dim, elementwise_affine=elementwise_affine))
            self._std_net = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=distribution_dim),
                std_activation)

        else:
            self._mean_net = nn.Linear(in_features=in_features, out_features=distribution_dim)
            self._std_net = nn.Sequential(nn.Linear(in_features=in_features, out_features=distribution_dim),
                                          std_activation)

    def forward(self, hidden_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mean_net(hidden_features), self._std_net(hidden_features)


class ConstantUncertaintyLayer(nn.Module):

    def __init__(self,
                 dim: int,
                 init_var: float,
                 min_var: float,
                 learn: bool):
        super(ConstantUncertaintyLayer, self).__init__()
        self._var_activation_layer = va.DiagGaussActivation(init_var=init_var,
                                                            min_var=min_var)

        if learn:
            self._raw_transition_var = nn.Parameter(torch.zeros(1, dim))
        else:
            _raw_transition_var = torch.zeros(1, dim)
            self.register_buffer(name="_raw_transition_var",
                                 tensor=_raw_transition_var)

    def forward(self, state_in: Union[torch.Tensor, int]) -> torch.Tensor:
        if isinstance(state_in, int):
            batch_size = state_in
        else:
            batch_size = state_in.shape[0]
        return self._var_activation_layer(self._raw_transition_var.repeat(batch_size, 1))


class Reshape(nn.Module):

    def __init__(self, shape: list[int]):
        """
        Reshape Layer
        :param shape: new shape of the tensor
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


class ConstantScalarUncertaintyNet(nn.Module):

    def __init__(self,
                 mean_net: nn.Module,
                 uncertainty_val: float = 1.0):

        super(ConstantScalarUncertaintyNet, self).__init__()
        self._mean_net = mean_net
        self.register_buffer(name="_uncertainty_val",
                             tensor=uncertainty_val * torch.ones(size=(1,)))

        # For potential time distribution
        self.num_outputs = 2
        self.td_copy_trough = [1]

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mean_net(input_features), self._uncertainty_val
