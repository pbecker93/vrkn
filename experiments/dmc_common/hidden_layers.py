from typing import Tuple
import torch
from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.common.modules import Reshape, ConstantScalarUncertaintyNet
import ssm_mbrl.common.dense_nets as dn


nn = torch.nn


def get_standard_ae_mujoco_factories(obs_type: str):
    if obs_type == "img":
        encoder_factories = [ImageEncoderFactory()]
        decoder_factories = [ImageDecoderFactory(),
                             dn.ConstantScalarUncertaintyDenseNetFactory()]
    elif obs_type == "img_pro_pos":
        encoder_factories = [ImageEncoderFactory(),
                             dn.HiddenMLPFactory()]
        decoder_factories = [ImageDecoderFactory(),
                             dn.ConstantScalarUncertaintyDenseNetFactory(),
                             dn.ConstantScalarUncertaintyDenseNetFactory()]

    else:
        raise AssertionError
    return encoder_factories, decoder_factories


class ImageEncoderFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = ConfigDict()
        config.base_depth = 32
        config.activation = "ReLU"
        config.mask_if_possible = True

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(input_size: tuple[int, int, int],
              config: ConfigDict) -> Tuple[nn.Module, int]:
        assert len(input_size) == 3
        assert input_size[0] == 3 and input_size[1] == 64 and input_size[2] == 64

        """Encoder hidden layers as used in Dreamer, Planet and "World Models" for mujoco_old data_gen"""
        return nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=config.base_depth,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            getattr(nn, config.activation)(),

            nn.Conv2d(in_channels=config.base_depth,
                      out_channels=2 * config.base_depth,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            getattr(nn, config.activation)(),

            nn.Conv2d(in_channels=2 * config.base_depth,
                      out_channels=4 * config.base_depth,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            getattr(nn, config.activation)(),

            nn.Conv2d(in_channels=4 * config.base_depth,
                      out_channels=8 * config.base_depth,
                      kernel_size=(4, 4),
                      padding=(0, 0),
                      stride=(2, 2)),
            getattr(nn, config.activation)(),

            nn.Flatten()), 4 * 8 * config.base_depth


class ImageDecoderFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = ConfigDict()
        config.base_depth = 32
        config.activation = "ReLU"
        config.output_std = 1.0
        config.mask_if_possible = True
        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(input_size: int,
              output_size: tuple[int, int, int],
              config: ConfigDict) -> nn.Module:
        assert len(output_size) == 3
        assert output_size[0] == 3 and output_size[1] == 64 and output_size[2] == 64

        mean_net = nn.Sequential(
            # h1
            nn.Linear(in_features=input_size,
                      out_features=4 * 8 * config.base_depth),
            #  nn.LeakyReLU() if leaky_relu else nn.ReLU(), -> no non-linearity here in dreamer implementation
            Reshape(shape=[-1, 4 * 8 * config.base_depth, 1, 1]),
            # h2
            nn.ConvTranspose2d(in_channels=4 * 8 * config.base_depth,
                               out_channels=4 * config.base_depth,
                               kernel_size=(5, 5),
                               stride=(2, 2)),
            getattr(nn, config.activation)(),

            # h3
            nn.ConvTranspose2d(in_channels=4 * config.base_depth,
                               out_channels=2 * config.base_depth,
                               kernel_size=(5, 5),
                               stride=(2, 2)),
            getattr(nn, config.activation)(),

            # h4
            nn.ConvTranspose2d(in_channels=2 * config.base_depth,
                               out_channels=1 * config.base_depth,
                               kernel_size=(6, 6),
                               stride=(2, 2)),
            getattr(nn, config.activation)(),

            # h5
            nn.ConvTranspose2d(in_channels=1 * config.base_depth,
                               out_channels=3,
                               kernel_size=(6, 6),
                               stride=(2, 2)),
        )
        return ConstantScalarUncertaintyNet(mean_net=mean_net, uncertainty_val=config.output_std)
