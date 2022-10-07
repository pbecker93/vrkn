import torch

from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.vrkn.transition.vrkn_tm import VRKNTM
from ssm_mbrl.vrkn.transition.baseline_transitions.vrkn_no_mcd_tm import VRKNNoMCDTM

nn = torch.nn


class TransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True):
        config = ConfigDict()

        config.type = "vrkn"

        config.lod = 230
        # Transition Model:
        config.hidden_size = 300
        config.activation = "ReLU"

        config.init_ev = 0.9
        config.min_ev = 0.1
        config.max_ev = 0.99

        config.init_trans_var = 0.01
        config.min_trans_var = 0.001
        config.max_trans_var = 0.1
        config.trans_var_sigmoid = True

        config.init_obs_var = 1.0
        config.min_obs_var = 0.01
        config.max_obs_var = 5.0  # ignored if not obs_var_sigmoid
        config.obs_var_sigmoid = False
        config.obs_output_normalization = "none"

        config.initial_state_var = 1.0
        config.learn_initial_state_var = True
        config.min_initial_state_var = 0.001

        config.mc_drop_prob = 0.1

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(config: ConfigDict,
              action_dim: int,
              obs_sizes: list[int]):
        common_kwargs = dict(obs_sizes=obs_sizes,
                             action_dim=action_dim,
                             lod=config.lod,
                             init_obs_var=config.init_obs_var,
                             min_obs_var=config.min_obs_var,
                             max_obs_var=config.max_obs_var,
                             obs_var_sigmoid=config.obs_var_sigmoid,
                             obs_output_normalization=config.obs_output_normalization,
                             hidden_size=config.hidden_size,
                             activation=config.activation,
                             init_ev=config.init_ev,
                             min_ev=config.min_ev,
                             max_ev=config.max_ev,
                             init_trans_var=config.init_trans_var,
                             trans_var_sigmoid=config.trans_var_sigmoid,
                             min_trans_var=config.min_trans_var,
                             max_trans_var=config.max_trans_var,
                             learn_initial_state_var=config.learn_initial_state_var,
                             initial_state_var=config.initial_state_var,
                             min_initial_state_var=config.min_initial_state_var)

        if config.type == "vrkn":
            return VRKNTM(**common_kwargs,
                          mc_drop_prob=config.mc_drop_prob)
        elif config.type == "vrkn_no_mcd":
            return VRKNNoMCDTM(**common_kwargs)
        else:
            raise ValueError(f"Unknown transition model type: {config.type}")




