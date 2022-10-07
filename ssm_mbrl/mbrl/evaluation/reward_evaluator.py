import torch
from typing import Optional
import pathlib
import os
from collections import OrderedDict

from ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy

from ssm_mbrl.common.bayesian_modules import MAPEstimate
from ssm_mbrl.mbrl.evaluation.video_logger import VideoLogger
from ssm_mbrl.mbrl.common.data_collector import PolicyEnvInterface


class RewardEvaluator(PolicyEnvInterface):

    def __init__(self,
                 env,
                 policy: AbstractPolicy,
                 num_eval_sequences: int,
                 use_map: bool,
                 eval_at_mean: bool,
                 max_sequence_length: int = -1,
                 eval_interval: int = 1,
                 record_eval_vid: bool = False,
                 log_info: Optional[dict] = None,
                 record_kwargs: Optional[dict] = None,
                 save_path: Optional[str] = None):
        super().__init__(policy=policy)

        self._env = env
        self._policy = policy
        self._num_sequences = num_eval_sequences
        self._use_map_for_eval = use_map
        self._max_sequence_length = max_sequence_length
        self._eval_at_mean = eval_at_mean
        self._eval_interval = eval_interval
        self._log_info = log_info
        render_kwargs = record_kwargs if record_kwargs is not None else {}
        self._rerender = render_kwargs.pop("rerender", False)
        self._obs_idx = render_kwargs.pop("obs_idx", 0)

        self._record_eval_vid = record_eval_vid
        if self._record_eval_vid:
            self._vid_logger = VideoLogger(save_path=None if save_path is None else os.path.join(save_path, "eval_vid"),
                                           to_wandb=False,
                                           render_kwargs=render_kwargs)
        else:
            self._vid_logger = None

        if save_path is not None:
            pathlib.Path(os.path.join(save_path, "eval_vid")).mkdir(parents=True, exist_ok=True)

    def _rollout_policy(self):

        all_infos = []

        obs, info = self._env.reset()
        all_infos.append(info)

        if self._vid_logger is not None:
            if self._rerender:
                self._vid_logger.record_env(self._env)
            else:
                self._vid_logger.record_obs(obs[self._obs_idx])
        action, policy_state = self._policy.get_initial(batch_size=1)

        done = False
        steps = 0
        episode_return = 0

        while not done and (self._max_sequence_length < 0 or steps < self._max_sequence_length):

            obs_for_pol = self._prepare_observation_for_policy(observation=obs)
            action_for_pol = self._prepare_action_for_policy(action=action)
            ov_for_pol = self._prepare_obs_valid_for_policy(info_dict=info)
            action_from_pol, policy_state = self._policy(observation=obs_for_pol,
                                                         prev_action=action_for_pol,
                                                         policy_state=policy_state,
                                                         sample=not self._eval_at_mean,
                                                         obs_valid=ov_for_pol)
            action = torch.squeeze(action_from_pol, dim=0).cpu()

            obs, reward, done, info = self._env.step(action=action)
            all_infos.append(info)
            if self._vid_logger is not None:
                if self._rerender:
                    self._vid_logger.record_env(self._env)
                else:
                    self._vid_logger.record_obs(obs[self._obs_idx])
            steps += 1
            episode_return += reward.detach().cpu().numpy().squeeze()
        return episode_return, steps, all_infos

    def _aggregate_infos(self, all_infos):
        if self._log_info is None:
            return OrderedDict()
        else:
            log_dict = OrderedDict()
            for k, v in self._log_info.items():
                if k in all_infos[0][0].keys():
                    all_values = []
                    agg_fn = getattr(torch, v.get("episode_agg", "mean"))
                    for episode_info in all_infos:
                        all_values.append(agg_fn(torch.stack([info[k] for info in episode_info])))
                    log_dict[k] = getattr(torch, v.get("agg", "mean"))(torch.stack(all_values)).cpu().numpy()
            return log_dict

    def evaluate(self,
                 iteration: int):
        def _evaluate():
            with torch.inference_mode():
                average_return = avg_length = 0.0
                all_infos = []
                for i in range(self._num_sequences):
                    if i == 0 and self._vid_logger is not None:
                        self._vid_logger.reset()
                        episode_return, episode_length, infos = self._rollout_policy()
                        self._vid_logger.save(file_name="eval_vid", step=iteration)
                    else:
                        episode_return, episode_length, infos = self._rollout_policy()

                    average_return += episode_return / self._num_sequences
                    avg_length += episode_length / self._num_sequences
                    all_infos.append(infos)
                log_dict = OrderedDict(average_reward=average_return, average_length=avg_length)
                log_dict |= self._aggregate_infos(all_infos)
                return log_dict

        if iteration % self._eval_interval == 0:
            if self._use_map_for_eval:
                with MAPEstimate(self._policy.model):
                    return _evaluate()
            else:
                return _evaluate()
        return None

    @staticmethod
    def name() -> str:
        return "reward_eval"


