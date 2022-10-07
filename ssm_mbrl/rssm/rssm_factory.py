import torch
from typing import Union


from ssm_mbrl.rssm.rssm import RSSM
from ssm_mbrl.rssm.smoothing_preprocessor import SmoothPreprocessor
from ssm_mbrl.rssm.transition.transition_factory import TransitionFactory
import ssm_mbrl.common.time_distribution as td

from ssm_mbrl.util.config_dict import ConfigDict

nn = torch.nn


class RSSMFactory:

    def __init__(self,
                 encoder_factories,
                 decoder_factories):
        self._encoder_factories = encoder_factories
        self._decoder_factories = decoder_factories

    @property
    def reward_decoder_idx(self) -> int:
        return len(self._decoder_factories) - 1

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:

        config = ConfigDict()
        config.smoothing_rssm = False

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

    def _build(self,
               config: ConfigDict,
               input_sizes: list[Union[int, tuple[int, int, int]]],
               output_sizes: list[Union[int, tuple[int, int, int]]],
               action_dim: int,
               with_obs_valid: bool):

        encoders, enc_out_sizes = [], []
        assert len(input_sizes) == len(self._encoder_factories)
        for i, (input_size, factory) in enumerate(zip(input_sizes, self._encoder_factories)):
            current_config = getattr(config, "encoder{}".format(i))
            enc, enc_out_size = factory.build(input_size=input_size,
                                              config=current_config)
            encoders.append(td.Jitted11TDPotentiallyMasked(base_module=enc,
                                                           mask_if_possible=current_config.mask_if_possible))
            enc_out_sizes.append(enc_out_size)

        if config.smoothing_rssm:
            smoothing_preprocessors = torch.nn.ModuleList()
            tm_obs_sizes = []
            for eos in enc_out_sizes:
                smoothing_preprocessors.append(SmoothPreprocessor(obs_dim=eos,
                                                                  num_layers=config.transition.num_layers,
                                                                  hidden_size=config.transition.hidden_size,
                                                                  activation=config.transition.activation,
                                                                  build_with_obs_vaild=with_obs_valid))
                tm_obs_sizes.append(config.transition.hidden_size)
        else:
            tm_obs_sizes = enc_out_sizes
            smoothing_preprocessors = None

        transition_model = TransitionFactory().build(config.transition,
                                                     obs_sizes=tm_obs_sizes,
                                                     action_dim=action_dim,
                                                     with_obs_valid=with_obs_valid)

        decoders = []
        assert len(output_sizes) == len(self._decoder_factories)
        for i, (output_size, factory) in enumerate(zip(output_sizes, self._decoder_factories)):
            current_config = getattr(config, "decoder{}".format(i))
            dec = factory.build(input_size=transition_model.feature_size,
                                output_size=output_size,
                                config=current_config)
            decoders.append(td.Jitted12TDPotentiallyMasked(base_module=dec,
                                                           mask_if_possible=current_config.mask_if_possible))

        return encoders, transition_model, decoders, smoothing_preprocessors

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              output_sizes: list[Union[int, tuple[int, int, int]]],
              action_dim: int,
              with_obs_valid: bool):
        encoders, transition_model, decoders, smoothing_preprocessors = self._build(config=config,
                                                                                    input_sizes=input_sizes,
                                                                                    output_sizes=output_sizes,
                                                                                    action_dim=action_dim,
                                                                                    with_obs_valid=with_obs_valid)
        return RSSM(encoders=torch.nn.ModuleList(encoders),
                    transition_model=transition_model,
                    decoders=torch.nn.ModuleList(decoders),
                    smoothing_preprocessors=smoothing_preprocessors)
