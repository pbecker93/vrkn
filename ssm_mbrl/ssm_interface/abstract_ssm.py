from typing import Optional, Callable
import torch

nn = torch.nn


class AbstractSSM(nn.Module):

    def encode(self,
               observations: list[torch.Tensor],
               obs_valid: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        raise NotImplementedError()

    @property
    def feature_size(self) -> int:
        raise NotImplementedError

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_initial_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_next_posterior(self,
                           observation: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict,
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict:
        raise NotImplementedError

    def predict_rewards_open_loop(self,
                                  initial_state: dict,
                                  actions: torch.Tensor):
        raise NotImplementedError

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        raise NotImplementedError

    def action_dim(self) -> int:
        raise NotImplementedError
