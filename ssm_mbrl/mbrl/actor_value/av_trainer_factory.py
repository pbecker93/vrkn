import torch

from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.mbrl.actor_value.av_policy import AVPolicy
from ssm_mbrl.mbrl.actor_value.av_trainer import AVPolicyTrainer
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM


class AVPolicyTrainerFactory:

    def __init__(self,
                 objective_factory):
        self._objective_factory = objective_factory

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.lambda_ = 0.95
        config.discount = 0.99

        config.model_learning_rate = 6e-4
        config.model_adam_epsilon = 1e-8
        config.model_clip_by_norm = True
        config.model_clip_norm = 100.0
        config.model_weight_decay = 0.0

        config.actor_learning_rate = 8e-5
        config.actor_adam_epsilon = 1e-8
        config.actor_clip_by_norm = True
        config.actor_clip_norm = 100.0
        config.actor_weight_decay = 0.0

        config.value_learning_rate = 8e-5
        config.value_adam_epsilon = 1e-8
        config.value_clip_by_norm = True
        config.value_clip_norm = 100.0
        config.value_weight_decay = 0.0

        config.imagine_horizon = 15
        config.imagine_from_smoothed = False

        config.entropy_bonus = 0.0

        config.add_subconf(name="objective",
                           sub_conf=self._objective_factory.get_default_config(finalize_adding=finalize_adding))

        config.eval_interval = -1

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              policy: AVPolicy,
              model: AbstractSSM,
              config: ConfigDict) -> AVPolicyTrainer:

        model_objective = self._objective_factory.build(model=model,
                                                        config=config.objective)

        return AVPolicyTrainer(policy=policy,
                               model_objective=model_objective,
                               lambda_=config.lambda_,
                               discount=config.discount,
                               imagine_horizon=config.imagine_horizon,
                               imagine_from_smoothed=config.imagine_from_smoothed,
                               model_learning_rate=config.model_learning_rate,
                               model_adam_eps=config.model_adam_epsilon,
                               model_clip_norm=config.model_clip_norm,
                               model_weight_decay=config.model_weight_decay,
                               actor_learning_rate=config.actor_learning_rate,
                               actor_adam_eps=config.actor_adam_epsilon,
                               actor_clip_norm=config.actor_clip_norm,
                               actor_weight_decay=config.actor_weight_decay,
                               value_learning_rate=config.value_learning_rate,
                               value_adam_eps=config.value_adam_epsilon,
                               value_clip_norm=config.value_clip_norm,
                               value_weight_decay=config.value_weight_decay,
                               entropy_bonus=config.entropy_bonus,
                               eval_interval=config.eval_interval)

