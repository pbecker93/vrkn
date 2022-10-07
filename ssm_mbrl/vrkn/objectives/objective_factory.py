from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
import ssm_mbrl.vrkn.objectives.autoencoding_vi_objective as aevi_obj


class AEVRKNObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.decoder_loss_scales = [1.0]

        config.kl_scale_factor = 1.0
        config.initial_kl_scale_factor = 0.0
        config.free_nats = 3.0

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(model,
              config: ConfigDict) -> AbstractObjective:

        return aevi_obj.AEVIObjective(vrkn=model,
                                      decoder_loss_scales=config.decoder_loss_scales,
                                      kl_scale_factor=config.kl_scale_factor,
                                      initial_kl_scale_factor=config.initial_kl_scale_factor,
                                      free_nats=config.free_nats)

