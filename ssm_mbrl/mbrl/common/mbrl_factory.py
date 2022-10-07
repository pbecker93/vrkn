import time
import torch

from ssm_mbrl.mbrl.common.data_collector import DataCollector
from ssm_mbrl.mbrl.common.wrappers import ObsNormalizationEnvWrapper
from ssm_mbrl.mbrl.common.replay_buffer import ReplayBuffer
from ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor

from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.util.stack_util import stack_maybe_nested_dicts

import warnings


class MBRLFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()

        config.add_subconf("initial_data_collection", ConfigDict())
        config.initial_data_collection.seq_length = -1
        config.initial_data_collection.num_sequences = 5

        config.add_subconf("data_collection", ConfigDict())
        config.data_collection.seq_length = -1
        config.data_collection.num_sequences = 1
        config.data_collection.action_noise_std = 0.3
        config.data_collection.use_map = False

        config.add_subconf("mbrl_exp", ConfigDict())
        config.mbrl_exp.normalize_obs = True
        config.mbrl_exp.model_updt_steps = 100
        config.mbrl_exp.model_updt_seq_length = 50
        config.mbrl_exp.model_updt_batch_size = 50

        config.add_subconf("img_preprocessing", ConfigDict())
        config.img_preprocessing.color_depth_bits = 5
        config.img_preprocessing.add_cb_noise = True

        config.add_subconf("validation", ConfigDict())
        config.validation.num_batches = 20
        config.validation.batch_size = 50

        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build_env_and_replay_buffer(env,
                                    img_preprocessor: ImgPreprocessor,
                                    config: ConfigDict):

        with torch.inference_mode():
            replay_buffer = ReplayBuffer(add_reward_to_obs=False,
                                         obs_are_images=env.obs_are_images,
                                         dataloader_num_workers=0,
                                         img_preprocessor=img_preprocessor)

            initial_obs, initial_acts, initial_rewards, initial_infos = \
                MBRLFactory.collect_initial_data(env=env,
                                                 num_sequences=config.initial_data_collection.num_sequences,
                                                 sequence_length=config.initial_data_collection.seq_length)
            if config.mbrl_exp.normalize_obs and not all(env.obs_are_images):
                obs_means, obs_stds = MBRLFactory._inplace_normalize(inputs=initial_obs,
                                                                     are_images=env.obs_are_images)
                env = ObsNormalizationEnvWrapper(env=env,
                                                 obs_means=obs_means,
                                                 obs_stds=obs_stds)
                replay_buffer.save_normalization_parameters(obs_means=obs_means,
                                                            obs_stds=obs_stds)
            replay_buffer.add_data(key="initial",
                                   observations=initial_obs,
                                   actions=initial_acts,
                                   rewards=initial_rewards,
                                   infos=initial_infos)
            return env, replay_buffer

    @staticmethod
    def build_data_collector(env,
                             policy: AbstractPolicy,
                             config: ConfigDict):
        return DataCollector(env=env,
                             policy=policy,
                             sequences_per_collect=config.data_collection.num_sequences,
                             max_sequence_length=config.data_collection.seq_length,
                             action_noise_std=config.data_collection.action_noise_std,
                             use_map=config.data_collection.use_map)

    @staticmethod
    def _inplace_normalize(inputs: list[list[torch.Tensor]],
                           are_images: list[bool]):
        means, stds = [], []
        for i in range(len(inputs[0])):
            if are_images[i]:
                means.append(None)
                stds.append(None)
            else:
                s, m = torch.std_mean(torch.cat([obs[i] for obs in inputs], dim=0), dim=0)
                if (s < 1e-2).any():
                    warnings.warn("Clipping std from below")
                    s = torch.maximum(s, 1e-2 * torch.ones_like(s))
                means.append(m)
                stds.append(s)
                [(inpt[i].sub_(m)).div_(s) for inpt in inputs]
        return means, stds

    @staticmethod
    def sample_action(action_space):
        eps = torch.rand(action_space.shape[0])
        return (action_space.high - action_space.low) * eps + action_space.low

    @staticmethod
    def _rollout(env,
                 sequence_length: int):
        observations, actions, rewards, infos = [], [], [], []
        obs, info = env.reset()
        observations.append(obs)
        actions.append(torch.zeros(env.action_space.shape[0]))
        rewards.append(torch.FloatTensor([0.0]))
        infos.append(info)
        done = False
        i = 0
        while not done and (sequence_length < 0 or i < sequence_length):
            action = MBRLFactory.sample_action(action_space=env.action_space)
            obs, reward, done, info = env.step(action=action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            i += 1
        return observations, actions, rewards, infos

    @staticmethod
    def collect_initial_data(env,
                             num_sequences: int,
                             sequence_length: int = -1):
        t0 = time.time()
        with torch.inference_mode():
            all_observations, all_actions, all_rewards, all_infos = [], [], [], []

            for i in range(num_sequences):
                observations, actions, rewards, infos = MBRLFactory._rollout(env=env, sequence_length=sequence_length)
                no = len(observations[0])
                all_observations.append([torch.stack([o[i] for o in observations], dim=0) for i in range(no)])
                all_actions.append(torch.stack(actions, dim=0))
                all_rewards.append(torch.stack(rewards, dim=0))
                all_infos.append(stack_maybe_nested_dicts(infos, dim=0))
            print("Random collection took", time.time() - t0)
            return all_observations, all_actions, all_rewards, all_infos
