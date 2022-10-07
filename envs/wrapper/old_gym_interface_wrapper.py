import gym
import numpy as np

# Software Engineering yaaay


class OldGymInterfaceWrapper:

    def __init__(self, base_env: gym.Env):
        self._base_env = base_env

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._base_env, name)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        return obs, reward, terminated or truncated, info
