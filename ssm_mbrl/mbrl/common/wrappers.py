import torch
import gym


class ObsNormalizationEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 obs_means: list[torch.Tensor],
                 obs_stds: list[torch.Tensor]):
        super(ObsNormalizationEnvWrapper, self).__init__(env)
        self._obs_means = obs_means
        self._obs_stds = obs_stds

    @staticmethod
    def _normalize_list(entries, means, stds):
        normalized_entries = []
        for o, m, s in zip(entries, means, stds):
            if m is not None and s is not None:
                normalized_entries.append((o - m) / s)
            else:
                normalized_entries.append(o)
        return normalized_entries

    def reset(self) -> tuple[list[torch.Tensor], dict]:
        observations, infos = self.env.reset()
        return self._normalize_list(entries=observations, means=self._obs_means, stds=self._obs_stds), infos

    def step(self, action: torch.Tensor) -> tuple[list[torch.Tensor], float, bool, dict]:
        observations, reward, done, infos = self.env.step(action)
        normalized_obs = self._normalize_list(entries=observations, means=self._obs_means, stds=self._obs_stds)
        return normalized_obs, reward, done, infos

    def de_normalize(self, obs: list[torch.Tensor]) -> list[torch.Tensor]:
        de_normalized_obs = []
        for o, m, s in zip(obs, self._obs_means, self._obs_stds):
            if m is not None and s is not None:
                de_normalized_obs.append(o * s + m)
        return de_normalized_obs
