import torch

from ssm_mbrl.util.config_dict import ConfigDict
import ssm_mbrl.common.modules as mod


nn = torch.nn


def build_hidden_layers(in_features: int,
                        layer_sizes: list[int],
                        activation: str = "ReLU") -> tuple[nn.ModuleList, int]:

    layers = []
    n_in = in_features
    n_out = n_in
    for layer_size in layer_sizes:
        n_out = layer_size
        layers.append(nn.Linear(in_features=n_in, out_features=n_out))
        layers.append(getattr(nn, activation)())
        n_in = n_out
    return nn.ModuleList(layers), n_out


class HiddenMLPFactory:

    @staticmethod
    def get_default_config(finalize_adding=False):
        config = ConfigDict()

        config.num_hidden = 3
        config.hidden_size = 300
        config.activation = "ELU"
        config.mask_if_possible = False
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(input_size: int, config: ConfigDict) -> tuple[nn.Module, int]:
        hidden_layers, hidden_output_size = \
            build_hidden_layers(in_features=input_size,
                                layer_sizes=[config.hidden_size] * config.num_hidden,
                                activation=config.activation)
        return nn.Sequential(*hidden_layers), hidden_output_size


class DenseNetFactory:

    @staticmethod
    def get_default_config(finalize_adding=False):
        config = ConfigDict()

        config.num_hidden = 3
        config.hidden_size = 300
        config.activation = "ELU"
        config.mask_if_possible = False
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(input_size: int, output_size: int, config: ConfigDict) -> nn.Module:
        layers, hidden_output_size = \
            build_hidden_layers(in_features=input_size,
                                layer_sizes=[config.hidden_size] * config.num_hidden,
                                activation=config.activation)
        layers.append(nn.Linear(in_features=hidden_output_size, out_features=output_size))
        return nn.Sequential(*layers)


class ConstantScalarUncertaintyDenseNetFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = DenseNetFactory.get_default_config(finalize_adding=False)
        config.output_std = 1.0
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(input_size: int,
              output_size: int,
              config: ConfigDict) -> nn.Module:
        mean_net = DenseNetFactory.build(input_size=input_size,
                                         output_size=output_size,
                                         config=config)
        return mod.ConstantScalarUncertaintyNet(mean_net=mean_net,
                                                uncertainty_val=config.output_std)


