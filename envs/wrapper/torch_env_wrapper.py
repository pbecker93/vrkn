import gym
import torch
import numpy as np


class TorchBox(gym.spaces.Box):

    def __init__(self,
                 low: torch.Tensor,
                 high=torch.Tensor,
                 shape=tuple,
                 dtype = torch.dtype):
        self.low = low
        self.high = high
        self._shape = shape
        self.dtype = dtype

    def __getattr__(self, item):
        if item not in ["shape"]:
            raise NotImplementedError("Probably not implemented")
        return self.__getattribute__(item)


class TorchTuple(gym.spaces.Tuple):

    def __init__(self,
                 spaces):
        self.spaces = tuple(spaces)

    def __getattr__(self, item):
        raise NotImplementedError("Probably not implemented")


class TorchEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 dtype: torch.dtype = torch.float32):
        super(TorchEnvWrapper, self).__init__(env)
        self._dtype = dtype
        self._obs_are_images = env.obs_are_images

        self.action_space = TorchBox(low=torch.from_numpy(env.action_space.low).to(self._dtype),
                                     high=torch.from_numpy(env.action_space.high).to(self._dtype),
                                     shape=env.action_space.shape,
                                     dtype=self._dtype)

        obs_spaces = []
        for o, is_image in zip(env.observation_space, self._obs_are_images):
            if is_image:
                assert o.dtype == np.uint8, "Images need to be uint8"
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low),
                                           high=torch.from_numpy(o.high),
                                           shape=(o.shape[2], o.shape[0], o.shape[1]),
                                           dtype=torch.uint8))
            else:
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low).to(self._dtype),
                                           high=torch.from_numpy(o.high).to(self._dtype),
                                           shape=o.shape,
                                           dtype=self._dtype))
        self.observation_space = TorchTuple(obs_spaces)

    def _obs_to_torch(self, obs: list[np.ndarray]) -> list[torch.Tensor]:
        torch_obs = []
        for o, is_image in zip(obs, self._obs_are_images):
            if is_image:
                torch_obs.append(torch.from_numpy(np.ascontiguousarray(np.transpose(o, axes=(2, 0, 1)))))
            else:
                torch_obs.append(torch.from_numpy(o).to(self._dtype))
        return torch_obs

    def _dict_to_torch(self, np_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        torch_dict = {}
        for k, v in np_dict.items():
            if k == "obs_valid":
                torch_dict[k] = [torch.BoolTensor([b]) for b in v]
            elif k == "loss_mask":
                torch_dict[k] = [torch.from_numpy(lm) for lm in v]
            elif np.isscalar(v):
                torch_dict[k] = torch.Tensor([v]).to(self._dtype)
            else:
                torch_dict[k] = torch.from_numpy(v).to(self._dtype)
        return torch_dict

    def reset(self):
        np_obs, np_info = self.env.reset()
        return self._obs_to_torch(np_obs), self._dict_to_torch(np_info)

    def step(self, action: torch.Tensor):
        np_action = action.detach().cpu().numpy().astype(self.env.action_space.dtype)
        np_obs, scalar_reward, done, np_info = self.env.step(action=np_action)
        reward = torch.Tensor([scalar_reward]).to(self._dtype)
        done = torch.Tensor([done]).to(torch.bool)
        return self._obs_to_torch(np_obs), reward, done, self._dict_to_torch(np_info)