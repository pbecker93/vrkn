import torch
import os
import numpy as np

from envs.wrapper.torch_env_wrapper import TorchEnvWrapper
from envs.wrapper.old_gym_interface_wrapper import OldGymInterfaceWrapper
from ssm_mbrl.util.config_dict import ConfigDict
from envs.dmc.suite_env import SuiteBaseEnv
from envs.dmc.dmc_mbrl_env import DMCMBRLEnv, ObsTypes
from envs.wrapper.lower_img_freq_wrapper import LowerImgFreqWrapper
from envs.wrapper.occlusion_wrapper import OcclusionWrapper


class DMCEnvFactory:

    SUBSAMPLE_MIN_SKIP = 4
    SUBSAMPLE_MAX_SKIP = 8

    DISK_OCCLUSIONS = {"temp_step": 5,
                       "folder": "disks"}

    WALL_OCCLUSIONS = {"temp_step": 1,
                       "folder": "walls"}

    OCCLUSION_DATA_PATH = os.path.join(os.path.dirname(__file__), "occlusion_data")

    OCCLUSIONS = {"disks": DISK_OCCLUSIONS,
                  "walls": WALL_OCCLUSIONS}

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        config.env = "cheetah_run"
        config.action_repeat = -1  # env default
        config.obs_type = "img"

        config.transition_noise_std = 0.0

        config.occluded = False
        config.occlusion_type = "disks"

        config.subsample_img_freq = False

        if finalize_adding:
            config.finalize_adding()

        return config

    @staticmethod
    def build(seed: int,
              config: ConfigDict,
              dtype=torch.float32):

        env_list = config.env.split("_")
        domain_name = ""
        for s in env_list[:-2]:
            domain_name += s + "_"
        domain_name += env_list[-2]
        task_name = env_list[-1]

        base_env = SuiteBaseEnv(domain_name=domain_name,
                                task_name=task_name,
                                seed=seed)

        obs_types = {"img": ObsTypes.IMAGE,
                     "img_pro_pos": ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION}
        env = DMCMBRLEnv(base_env=base_env,
                         seed=seed,
                         action_repeat=config.action_repeat,
                         obs_type=obs_types[config.obs_type],
                         transition_noise_std=config.transition_noise_std,
                         img_size=(64, 64),
                         image_to_info=False)

        if config.occluded:
            occlusion_kwargs = DMCEnvFactory.OCCLUSIONS[config.occlusion_type]
            env = OcclusionWrapper(env=env,
                                   temp_step=occlusion_kwargs["temp_step"],
                                   vid_folder=os.path.join(DMCEnvFactory.OCCLUSION_DATA_PATH,
                                                           occlusion_kwargs["folder"]),
                                   occluded_img_idx=0)

        if config.subsample_img_freq:
            assert np.sum(env.obs_are_images) == 1, "Only implemented when exactly 1 obs is image"
            env = LowerImgFreqWrapper(env=env,
                                      seed=seed,
                                      min_skip=DMCEnvFactory.SUBSAMPLE_MIN_SKIP,
                                      max_skip=DMCEnvFactory.SUBSAMPLE_MAX_SKIP,
                                      img_idx=int(np.argmax(env.obs_are_images)))

        env = OldGymInterfaceWrapper(env)
        env = TorchEnvWrapper(env, dtype)
        return env

