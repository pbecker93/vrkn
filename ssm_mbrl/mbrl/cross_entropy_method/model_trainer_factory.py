import torch

from ssm_mbrl.mbrl.cross_entropy_method.model_trainer import ModelTrainer
from ssm_mbrl.mbrl.cross_entropy_method.cem_policy import CEMPolicy
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_mbrl.util.config_dict import ConfigDict


class ModelTrainerFactory:

    def __init__(self,
                 objective_factory):
        self._objective_factory = objective_factory

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.learning_rate = 1e-3
        config.adam_epsilon = 1e-8
        config.clip_grad_by_norm = True
        config.clip_norm = 1000.0

        config.eval_interval = 1

        config.add_subconf(name="objective",
                           sub_conf=self._objective_factory.get_default_config(finalize_adding=finalize_adding))

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              policy: CEMPolicy,
              model: AbstractSSM,
              config: ConfigDict) -> ModelTrainer:

        assert isinstance(policy, CEMPolicy)
        objective = self._objective_factory.build(model=model,
                                                  config=config.objective)

        return ModelTrainer(model=model,
                            objective=objective,
                            learning_rate=config.learning_rate,
                            adam_epsilon=config.adam_epsilon,
                            clip_grad_by_norm=config.clip_grad_by_norm,
                            clip_norm=config.clip_norm,
                            eval_interval=config.eval_interval)
