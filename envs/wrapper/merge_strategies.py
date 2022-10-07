
import numpy as np

from ..distractors import DistractionSource


class BaseStrategy:
    def __init__(self, source: DistractionSource, intensity=1):
        self.source = source
        self.intensity = intensity

    def merge(self, obs: np.array) -> np.array:
        raise NotImplementedError

    def merge_timeseries(self, obs: np.array) -> np.array:
        """
        Used for offline adding of observations
        :param obs:
        :return:
        """
        self.source.reset()
        augmented_obs = []
        for timestep in obs:
            augmented_obs.append(self.merge(timestep))
        return np.array(augmented_obs)

    def get_last_mask(self):
        raise NotImplementedError


class FrontMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        augmented_obs = np.copy(obs)
        augmented_obs[mask] = img[mask]
        self._mask = mask
        return augmented_obs

    def get_last_mask(self):
        return self._mask


strategies = {"foreground": FrontMerge}
