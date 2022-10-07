from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.rssm.transition.r_rssm_tm import RRSSMTM
from ssm_mbrl.rssm.transition.baseline_transitions.s_rssm_tm import SRSSMTM
from ssm_mbrl.rssm.transition.baseline_transitions.mcd_rssm_tm import \
    McDropoutRRSSMTM, McDropoutSRSSMTM, FullMCDropoutRRSSMTM
# TODO Closed form transition models?


class TransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = False) -> ConfigDict:
        config = ConfigDict()
        config.type = "r_rssm"
        config.lsd = 30
        config.rec_state_dim = 200
        config.num_layers = 1
        config.hidden_size = 300
        config.min_std = 0.1
        config.activation = "ELU"
        config.state_part_for_update = "d"

        # Monte Carlo Dropout
        config.mc_drop_prob = 0.1

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict,
              obs_sizes: list[int],
              action_dim: int,
              with_obs_valid: bool):
        default_params = {"obs_sizes": obs_sizes,
                          "build_with_obs_valid": with_obs_valid,
                          "state_dim": config.lsd,
                          "action_dim": action_dim,
                          "num_layers": config.num_layers,
                          "hidden_size": config.hidden_size,
                          "min_std": config.min_std,
                          "activation": config.activation}

        if "r_rssm" in config.type:
            default_params |= {"rec_state_dim": config.rec_state_dim,
                               "state_part_for_update": config.state_part_for_update}
            if config.type == "r_rssm":
                return RRSSMTM(**default_params)
            elif config.type == "mc_r_rssm":
                return McDropoutRRSSMTM(mc_drop_prob=config.mc_drop_prob,
                                        **default_params)
            elif config.type == "full_mc_r_rssm":
                return FullMCDropoutRRSSMTM(mc_drop_prob=config.mc_drop_prob,
                                            **default_params)
            else:
                raise AssertionError
        elif "s_rssm" in config.type:
            if config.type == "s_rssm":
                return SRSSMTM(**default_params)
            elif config.type == "mc_s_rssm":
                return McDropoutSRSSMTM(mc_drop_prob=config.mc_drop_prob,
                                        **default_params)
        else:
            raise AssertionError
