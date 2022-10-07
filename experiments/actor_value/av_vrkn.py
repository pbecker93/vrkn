
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
from ssm_mbrl.mbrl.actor_value.av_policy_factory import ACPolicyFactory
from ssm_mbrl.mbrl.actor_value.av_trainer_factory import AVPolicyTrainerFactory
from ssm_mbrl.mbrl.evaluation.reward_eval_factory import RewardEvalFactory

nn = torch.nn

if __name__ == "__main__":
    seed = 0

    env_config = DMCEnvFactory().get_default_config()
    env_config.action_repeat = 2

    "Configure Env"
    env_config.env = "cheetah_run"
    # "img_pro_pos" for Sensor Fusion, "img" for all other
    env_config.obs_type = "img"
    # Set Transition Noise STD
    env_config.transition_noise_std = 0.0


    # For Occlusion Experiments
    env_config.occluded = False
    env_config.occlusion_type = "walls"  # "disks" or "walls"

    # True for missing observations and sensor fusion experiments
    env_config.subsample_img_freq = False

    encoder_factories, decoder_factories = dmc_archs.get_standard_ae_mujoco_factories(obs_type=env_config.obs_type)
    model_factory = VRKNFactory(encoder_factories=encoder_factories, decoder_factories=decoder_factories)
    """Configure VRKN (and Baselines) """
    model_config = model_factory.get_default_config()
    # "vrkn" for normal VRKN, "vrkn_no_mcd" for vrkn without epistemic uncertainty (VRKN (no MCD) in paper)
    model_config.transition.type = "vrkn"

    policy_config = ACPolicyFactory().get_default_config()
    policy_config.critic.activation = "Tanh"

    trainer_factory = AVPolicyTrainerFactory(AEVRKNObjectiveFactory())
    trainer_config = trainer_factory.get_default_config()
    trainer_config.objective.decoder_loss_scales = [1.0] * (1 if env_config.obs_type == "img" else 2) + [10.0]

    mbrl_config = MBRLFactory().get_default_config()

    reward_eval_config = RewardEvalFactory().get_default_config()

    experiment = MBRLExperiment(env_factory=DMCEnvFactory,
                                model_factory=model_factory,
                                policy_factory=ACPolicyFactory(),
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


