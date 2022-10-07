import torch
from typing import Union

import ssm_mbrl.common.time_distribution as td
from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.vrkn.vrkn import VRKN
from ssm_mbrl.vrkn.transition.transition_factory import TransitionFactory

nn = torch.nn


class VRKNFactory:

    def __init__(self,
                 encoder_factories,
                 decoder_factories):
        self._encoder_factories = encoder_factories
        self._decoder_factories = decoder_factories

    @property
    def reward_decoder_idx(self) -> int:
        return len(self._decoder_factories) - 1

    def get_default_config(self, finalize_adding: bool = True):

        config = ConfigDict()

        for i, factory in enumerate(self._encoder_factories):
            config.add_subconf(name="encoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))

        for i, factory in enumerate(self._decoder_factories):
            config.add_subconf(name="decoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))

        config.add_subconf(name="transition",
                           sub_conf=TransitionFactory.get_default_config(finalize_adding=finalize_adding))

        if finalize_adding:
            config.finalize_adding()

        return config

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              output_sizes: list[Union[int, tuple[int, int, int]]],
              action_dim: int,
              with_obs_valid: bool):

        encoders, enc_out_sizes = nn.ModuleList(), []
        assert len(input_sizes) == len(self._encoder_factories)
        for i, (input_size, factory) in enumerate(zip(input_sizes, self._encoder_factories)):
            cur_conf = getattr(config, "encoder{}".format(i))
            enc, enc_out_size = factory.build(input_size=input_size,
                                              config=cur_conf)
            encoders.append(td.Jitted11TDPotentiallyMasked(base_module=enc,
                                                           mask_if_possible=cur_conf.get("mask_if_possible", True)))
            enc_out_sizes.append(enc_out_size)

        transition_model = TransitionFactory.build(config.transition,
                                                   obs_sizes=enc_out_sizes,
                                                   action_dim=action_dim)

        decoders = nn.ModuleList()
        assert len(output_sizes) == len(self._decoder_factories)
        for i, (output_size, factory) in enumerate(zip(output_sizes, self._decoder_factories)):
            cur_conf = getattr(config, "decoder{}".format(i))
            dec = factory.build(input_size=transition_model.feature_size,
                                output_size=output_size,
                                config=cur_conf)
            decoders.append(td.Jitted12TDPotentiallyMasked(base_module=dec,
                                                           mask_if_possible=cur_conf.get("mask_if_possible", True)))

        return VRKN(encoders=encoders,
                    transition_model=transition_model,
                    decoders=decoders)

