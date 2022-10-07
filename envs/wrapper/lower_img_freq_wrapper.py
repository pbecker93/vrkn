import numpy as np
import gym
from typing import Union


class LowerImgFreqWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 seed: int,
                 min_skip: int,
                 max_skip: int,
                 img_idx: int = 0):
        super(LowerImgFreqWrapper, self).__init__(env=env)
        self._rng = np.random.RandomState(seed)
        self._min_skip = min_skip
        self._max_skip = max_skip
        self._next_valid_img_step = -1
        self._step = 0
        self._img_idx = img_idx

    def _reset_wrapper(self):
        self._step = 0
        self._next_valid_img_step = self._step + self._rng.randint(low=self._min_skip, high=self._max_skip + 1)

    def reset(self, **kwargs) -> Union[list[np.ndarray], tuple[gym.core.ObsType, dict]]:
        self._reset_wrapper()
        obs, info = self.env.reset(**kwargs)
        obs_valid = [True for _ in obs]
        info["obs_valid"] = obs_valid
        return obs, info

    def step(self, action: np.ndarray) -> tuple[list[np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_valid = [True for _ in obs]
        self._step += 1
        if self._step == self._next_valid_img_step:
            self._next_valid_img_step = self._step + self._rng.randint(low=self._min_skip, high=self._max_skip + 1)
        else:
            obs[self._img_idx] = np.zeros_like(obs[self._img_idx])
            obs_valid[self._img_idx] = False
        info["obs_valid"] = obs_valid

        return obs, reward, terminated, truncated, info
