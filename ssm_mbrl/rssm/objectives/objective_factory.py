from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.rssm.objectives.autoencoding_vi_objective import AEVIObjective


class AERSSMObjectiveFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.decoder_loss_scales = [1.0]

        config.kl_scale_factor = 1.0
        config.free_nats = 3.0

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(model,
              config: ConfigDict):
        return AEVIObjective(rssm=model,
                             decoder_loss_scales=config.decoder_loss_scales,
                             kl_scale_factor=config.kl_scale_factor,
                             free_nats=config.free_nats)
