import torch

from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.mbrl.actor_value.av_policy import Actor, ValueFn, AVPolicy
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor


class ACPolicyFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.mean_for_actor = False

        config.add_subconf("actor", ConfigDict())
        config.actor.num_hidden = 3
        config.actor.hidden_size = 300
        config.actor.activation = "ELU"
        config.actor.min_std = 1e-4
        config.actor.init_std = 5.0
        config.actor.mean_scale = 5.0

        config.add_subconf("critic", ConfigDict())
        config.critic.num_hidden = 3
        config.critic.hidden_size = 300
        config.critic.activation = "ELU"

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

        actor = Actor(in_dim=model.feature_size,
                      action_dim=action_space.shape[0],
                      num_layers=config.actor.num_hidden,
                      hidden_size=config.actor.hidden_size,
                      init_std=config.actor.init_std,
                      min_std=config.actor.min_std,
                      min_action=action_space.low,
                      max_action=action_space.high,
                      mean_scale=config.actor.mean_scale,
                      activation=config.actor.activation).to(device)

        critic = ValueFn(in_dim=model.feature_size,
                         num_layers=config.critic.num_hidden,
                         hidden_size=config.critic.hidden_size,
                         activation=config.critic.activation).to(device)

        return AVPolicy(model=model,
                        actor=actor,
                        critic=critic,
                        action_dim=action_space.shape[0],
                        obs_are_images=obs_are_images,
                        img_preprocessor=img_preprocessor,
                        device=device,
                        mean_for_actor=config.mean_for_actor)
