import numpy as np
from collections import namedtuple
import os
from typing import Optional


class _Cheetah:

    PROPRIOCEPTIVE_POS_SIZE = 6
    PROPRIOCEPTIVE_VEL_SIZE = 6
    POSITION_SIZE = 8
    DEFAULT_ACTION_REPEAT = 4

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][-6:]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][-6:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]


class _Walker:

    PROPRIOCEPTIVE_POS_SIZE = 14
    PROPRIOCEPTIVE_VEL_SIZE = 6
    POSITION_SIZE = 15
    DEFAULT_ACTION_REPEAT = 2

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["orientations"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][3:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["orientations"], np.array([state.observation["height"]])])


class _BallInCup:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 4
    DEFAULT_ACTION_REPEAT = 4

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:2]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:2]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]


class _Reacher:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 4
    DEFAULT_ACTION_REPEAT = 4

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["to_target"]])


class _Finger:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 2
    POSITION_SIZE = 6
    DEFAULT_ACTION_REPEAT = 2

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:2]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:2]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["touch"]])


class _Cartpole:

    PROPRIOCEPTIVE_POS_SIZE = 1
    PROPRIOCEPTIVE_VEL_SIZE = 1
    POSITION_SIZE = 3
    DEFAULT_ACTION_REPEAT = 8

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["position"][:1]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][:1]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["position"]


class _Pendulum:

    PROPRIOCEPTIVE_POS_SIZE = 2
    PROPRIOCEPTIVE_VEL_SIZE = 1
    POSITION_SIZE = 2
    DEFAULT_ACTION_REPEAT = 2

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["orientation"]

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return np.concatenate([state.observation["orientation"], state.observation["velocity"]])

    @staticmethod
    def get_position(state) -> np.ndarray:
        return state.observation["orientation"]


class _Hopper:

    PROPRIOCEPTIVE_POS_SIZE = 6
    PROPRIOCEPTIVE_VEL_SIZE = 4
    POSITION_SIZE = 8
    DEFAULT_ACTION_REPEAT = 2

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"][-4:], state.observation["touch"]])

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["velocity"][-4:]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["position"], state.observation["touch"]])


class _Quadruped:

    PROPRIOCEPTIVE_POS_SIZE = -1
    PROPRIOCEPTIVE_VEL_SIZE = -1
    POSITION_SIZE = -1
    DEFAULT_ACTION_REPEAT = 2

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_position(state) -> np.ndarray:
        raise NotImplementedError


class _Manipulator:

    PROPRIOCEPTIVE_POS_SIZE = 16
    PROPRIOCEPTIVE_VEL_SIZE = 8
    POSITION_SIZE = 33
    DEFAULT_ACTION_REPEAT = 1

    @staticmethod
    def get_proprioceptive_position(state) -> np.ndarray:
        return state.observation["arm_pos"].ravel()

    @staticmethod
    def get_proprioceptive_velocity(state) -> np.ndarray:
        return state.observation["arm_vel"]

    @staticmethod
    def get_position(state) -> np.ndarray:
        return np.concatenate([state.observation["arm_pos"].ravel(),     # 16
                               state.observation["touch"],               # 5
                               state.observation["hand_pos"],            # 4
                               state.observation["object_pos"],          # 4
                               state.observation["target_pos"]])         # 4


class SuiteBaseEnv:

    DMCEnvSpec = namedtuple("DMCEnvSpec", "domain_name task_name env_cls")
    DMC_ENV_CLASSES = {"cheetah": _Cheetah,
                       "walker": _Walker,
                       "ball_in_cup": _BallInCup,
                       "reacher": _Reacher,
                       "finger": _Finger,
                       "cartpole": _Cartpole,
                       "pendulum": _Pendulum,
                       "hopper": _Hopper,
                       "quadruped": _Quadruped,
                       "manipulator": _Manipulator}

    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 seed: int):

        os.environ["MUJOCO_GL"] = "egl"
        from dm_control import suite

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs={"random": seed})
        self._env_cls = self.DMC_ENV_CLASSES[domain_name]

        self.step = self._env.step
        self.reset = self._env.reset
        self.action_spec = self._env.action_spec
        self.observation_spec = self._env.observation_spec

    def reset(self):
        return self._env.reset()

    def step(self, action: np.ndarray):
        return self._env.step(action)

    @property
    def default_action_repeat(self) -> int:
        return self._env_cls.DEFAULT_ACTION_REPEAT

    def render(self, img_size: tuple[int, int], cam: Optional[str] = None) -> np.ndarray:
        return self._env.physics.render(camera_id=0 if cam is None or cam == "default" else cam,
                                        height=img_size[0],
                                        width=img_size[1])

    @property
    def proprioceptive_pos_size(self):
        return self._env_cls.PROPRIOCEPTIVE_POS_SIZE

    @property
    def proprioceptive_vel_size(self):
        return self._env_cls.PROPRIOCEPTIVE_VEL_SIZE

    @property
    def position_size(self):
        return self._env_cls.POSITION_SIZE

    @staticmethod
    def get_info(state) -> dict:
        return {}

    def get_position(self, state) -> np.ndarray:
        return self._env_cls.get_position(state)

    def get_proprioceptive_position(self, state) -> np.ndarray:
        return self._env_cls.get_proprioceptive_position(state)

    def get_proprioceptive_velocity(self, state) -> np.ndarray:
        return self._env_cls.get_proprioceptive_velocity(state)

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    @property
    def physics(self):
        return self._env.physics
