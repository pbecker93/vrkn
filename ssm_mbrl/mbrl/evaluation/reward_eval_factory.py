
from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from ssm_mbrl.mbrl.evaluation.reward_evaluator import RewardEvaluator


class RewardEvalFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.num_sequences = 10
        config.use_map = True
        config.eval_at_mean = True
        config.max_sequence_length = -1
        config.record_vid = False
        config.save_path = None
        config.eval_interval = 20
        config.log_info = None
        config.render_kwargs = None

        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              env,
              policy: AbstractPolicy,
              config: ConfigDict):
        return RewardEvaluator(env=env,
                               policy=policy,
                               num_eval_sequences=config.num_sequences,
                               use_map=config.use_map,
                               eval_at_mean=config.eval_at_mean,
                               max_sequence_length=config.max_sequence_length,
                               log_info=config.log_info,
                               eval_interval=config.eval_interval,
                               record_eval_vid=config.record_vid,
                               save_path=config.save_path,
                               record_kwargs=config.render_kwargs.copy() if config.render_kwargs is not None else None)

    @staticmethod
    def name() -> str:
        return RewardEvaluator.name()
