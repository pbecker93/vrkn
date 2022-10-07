import torch
from ssm_mbrl.mbrl.cross_entropy_method.cem_policy import CEMPolicy
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_mbrl.util.config_dict import ConfigDict


class CEMPolicyFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.initial_action_mean = 0.0
        config.initial_action_std = 1.0
        config.num_samples = 1000
        config.opt_steps = 10
        config.planning_horizon = 12
        config.take_best_k = 100

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(model: AbstractSSM,
              obs_are_images: list[bool],
              img_preprocessor: ImgPreprocessor,
              config: ConfigDict,
              action_space,
              device: torch.device):

        return CEMPolicy(model=model,
                         obs_are_images=obs_are_images,
                         action_dim=action_space.shape[0],
                         min_action=action_space.low,
                         max_action=action_space.high,
                         img_preprocessor=img_preprocessor,
                         device=device,
                         initial_action_mean=config.initial_action_mean,
                         initial_action_std=config.initial_action_std,
                         num_samples=config.num_samples,
                         opt_steps=config.opt_steps,
                         planning_horizon=config.planning_horizon,
                         take_best_k=config.take_best_k)
