import torch
from typing import Optional
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM

class AbstractPolicy:

    def __init__(self,
                 model: AbstractSSM,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: ImgPreprocessor,
                 device: torch.device):
        super(AbstractPolicy, self).__init__()
        self.model = model
        self._action_dim = action_dim
        self._obs_are_images = obs_are_images
        self._img_preprocessor = img_preprocessor
        self._device = device

    def __call__(self,
                 observation: list[torch.Tensor],
                 prev_action: torch.Tensor,
                 policy_state: dict,
                 sample: bool,
                 obs_valid: Optional[list[torch.Tensor]]) -> tuple[torch.Tensor, dict]:
        for i, obs in enumerate(observation):
            if self._obs_are_images[i]:
                observation[i] = self._img_preprocessor(obs)
        return self._call_internal(observation=observation,
                                   prev_action=prev_action,
                                   policy_state=policy_state,
                                   sample=sample,
                                   obs_valid=obs_valid)

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool,
                       obs_valid: Optional[list[torch.Tensor]]) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def get_initial(self, batch_size: int) -> tuple[torch.Tensor, dict]:
        post_state = self.model.get_initial_state(batch_size=batch_size)
        initial_action = torch.zeros(size=(batch_size, self.action_dim),
                                     device=next(iter(post_state.values())).device)
        return initial_action, post_state

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def action_dim(self) -> int:
        return self._action_dim
