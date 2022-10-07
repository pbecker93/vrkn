from typing import Union, Optional
import gym
from enum import Enum, EnumMeta
import numpy as np


class ObsTypes(str, Enum, metaclass=EnumMeta):
    IMAGE = "image"
    IMAGE_PROPRIOCEPTIVE_POSITION = "image_proprioceptive_position"


class _WhiteNoise:

    def __init__(self,
                 mu: np.ndarray,
                 sigma: np.ndarray):
        self._mu = mu
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(size=self._mu.shape) * self._sigma

    def reset(self):
        pass


class DMCMBRLEnv(gym.Env):

    SUPPORTED_OBS_TYPES = [ObsTypes.IMAGE,
                           ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION]

    def __init__(self,
                 base_env,
                 seed: int,
                 obs_type,
                 action_repeat: int = -1,
                 transition_noise_std: float = 0.0,
                 img_size: tuple[int, int] = (64, 64),
                 image_to_info: bool = False):

        super(DMCMBRLEnv, self).__init__()
        assert obs_type in self.SUPPORTED_OBS_TYPES, f"Unsupported observation type: {obs_type}"

        self._seed = seed
        self._base_env = base_env
        self._obs_type = obs_type
        self._action_repeat = self._base_env.default_action_repeat if action_repeat < 0 else action_repeat
        self._transition_noise_std = transition_noise_std
        self._img_size = img_size
        self._current_step = 0
        self._image_to_info = image_to_info
        self._fig = None

        self.action_space.seed(seed=seed)
        [os.seed(seed) for os in self.observation_space]

        if transition_noise_std <= 0.0:
            self._transition_noise_generator = None
        else:
            self._transition_noise_generator = \
                _WhiteNoise(mu=np.zeros(self.action_space.shape),
                            sigma=transition_noise_std * np.ones(self.action_space.shape))

    def _get_obs(self, state) -> list[np.ndarray]:
        if self._obs_type == ObsTypes.IMAGE:
            return [self._base_env.render(self._img_size)]
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION:
            return [self._base_env.render(self._img_size), self._base_env.get_proprioceptive_position(state=state)]

        else:
            raise AssertionError

    def _get_info(self, state):
        info = self._base_env.get_info(state)
        if self._image_to_info:
            return info | {"image": self._base_env.render(self._img_size)}
        else:
            return info

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = True,
              options: Optional[dict] = None) -> Union[gym.core.ObsType, tuple[gym.core.ObsType, dict]]:
        assert return_info, "Always call reset with return_info=True for MBRL environments"
        if self._transition_noise_generator is not None:
            self._transition_noise_generator.reset()
        state = self._base_env.reset()
        self._current_step = 0
        return self._get_obs(state), self._get_info(state)

    def step(self, action: np.ndarray) -> tuple[gym.core.ObsType, float, bool, bool, dict]:
        if self._transition_noise_generator is not None:
            action = action + self._transition_noise_generator()
            action = np.clip(a=action, a_min=self.action_space.low, a_max=self.action_space.high)
        reward = 0
        self._current_step += 1
        truncated = False
        state = None
        for k in range(self._action_repeat):
            state = self._base_env.step(action)
            reward += state.reward
            truncated = state.last()
            if truncated:
                break
        return self._get_obs(state), reward, False, truncated, self._get_info(state)

    @property
    def obs_are_images(self) -> list[bool]:
        if self._obs_type == ObsTypes.IMAGE:
            return [True]
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION:
            return [True, False]
        else:
            raise AssertionError

    @property
    def action_dim(self) -> int:
        return self._base_env.action_spec().shape[0]

    @staticmethod
    def _get_ld_space(dim):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=float)

    def _get_img_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self._img_size[0], self._img_size[1], 3), dtype=np.uint8)

    @property
    def observation_space(self):
        if self._obs_type == ObsTypes.IMAGE:
            return gym.spaces.Tuple([self._get_img_space()])
        elif self._obs_type == ObsTypes.IMAGE_PROPRIOCEPTIVE_POSITION:
            ld_space = self._get_ld_space(self._base_env.proprioceptive_pos_size)
            return [self._get_img_space(), ld_space]
        else:
            raise AssertionError

    @property
    def state_space(self):
        return NotImplemented

    @property
    def action_space(self):
        return gym.spaces.Box(low=self._base_env.action_spec().minimum,
                              high=self._base_env.action_spec().maximum,
                              shape=self._base_env.action_spec().shape,
                              dtype=float)

    @property
    def max_seq_length(self) -> int:
        return 1000 // self._action_repeat

    def render(self, *args, **kwargs) -> np.ndarray:
        if len(args) == 0 and "img_size" not in kwargs.keys():
            kwargs["img_size"] = self._img_size
        return self._base_env.render(*args, **kwargs)
