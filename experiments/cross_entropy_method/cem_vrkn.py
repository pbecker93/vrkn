import os
#os.environ["PYTORCH_JIT"] = "0"
os.environ["MUJOCO_GL"] = "egl"

import torch

import experiments.dmc_common.hidden_layers as dmc_archs
from envs.env_factory import DMCEnvFactory

from ssm_mbrl.vrkn.vrkn_factory import VRKNFactory
from ssm_mbrl.vrkn.objectives.objective_factory import AEVRKNObjectiveFactory
from ssm_mbrl.mbrl.common.mbrl_experiment import MBRLExperiment
from ssm_mbrl.mbrl.common.mbrl_factory import MBRLFactory
from ssm_mbrl.mbrl.cross_entropy_method.cem_policy_factory import CEMPolicyFactory
from ssm_mbrl.mbrl.cross_entropy_method.model_trainer_factory import ModelTrainerFactory
from ssm_mbrl.mbrl.evaluation.reward_eval_factory import RewardEvalFactory

nn = torch.nn

if __name__ == "__main__":
    seed = 0

    env_config = DMCEnvFactory().get_default_config()
    env_config.action_repeat = -1  # default for planet envs (see Hafner et al 2019)

    "Configure Env"
    env_config.env = "cheetah_run"

    encoder_factories, decoder_factories = dmc_archs.get_standard_ae_mujoco_factories(obs_type="img")
    model_factory = VRKNFactory(encoder_factories=encoder_factories, decoder_factories=decoder_factories)
    """Configure VRKN (and Baselines) """
    model_config = model_factory.get_default_config()
    # r_rssm for RSSM, mcd_r_rssm for MCD-RSSM, "s_rssm" for "Stochastic State Model" form Hafner et al 2019
    model_config.transition.type = "vrkn"
    # False for "normal" RSSM (MCD-RSSM) , True for "smooth" RSSM (MCD-RSSM)

    model_config.transition.hidden_size = 200
    model_config.transition.activation = "ELU"
    model_config.decoder1.activation = "ReLU"
    model_config.decoder1.hidden_size = 200
    model_config.decoder1.num_hidden = 2

    policy_config = CEMPolicyFactory().get_default_config()

    trainer_factory = ModelTrainerFactory(AEVRKNObjectiveFactory())
    trainer_config = trainer_factory.get_default_config()
    trainer_config.objective.decoder_loss_scales = [1.0, 10.0]

    mbrl_config = MBRLFactory().get_default_config()

    reward_eval_config = RewardEvalFactory().get_default_config()

    experiment = MBRLExperiment(env_factory=DMCEnvFactory,
                                model_factory=model_factory,
                                policy_factory=CEMPolicyFactory(),
                                trainer_factory=trainer_factory,
                                mbrl_factory=MBRLFactory(),
                                policy_eval_factories=[RewardEvalFactory()],
                                verbose=True,
                                seed=seed,
                                use_cuda_if_available=True,
                                fully_deterministic=False)

    experiment.build(env_config=env_config,
                     model_config=model_config,
                     policy_config=policy_config,
                     trainer_config=trainer_config,
                     mbrl_config=mbrl_config,
                     policy_eval_config={"reward_eval": reward_eval_config})

    for i in range(1001):
        log_dict = experiment.iterate(i)

