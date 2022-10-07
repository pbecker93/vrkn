import torch
from typing import Optional

from ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
dists = torch.distributions


class CEMPolicy(AbstractPolicy):

    def __init__(self,
                 model: AbstractSSM,
                 action_dim: int,
                 min_action: torch.Tensor,
                 max_action: torch.Tensor,
                 device: torch.device,
                 obs_are_images: list[bool],
                 img_preprocessor: ImgPreprocessor,
                 initial_action_mean: float,
                 initial_action_std: float,
                 num_samples: int,
                 opt_steps: int,
                 planning_horizon: int,
                 take_best_k: int):

        super(CEMPolicy, self).__init__(model=model,
                                        action_dim=action_dim,
                                        obs_are_images=obs_are_images,
                                        img_preprocessor=img_preprocessor,
                                        device=device)

        action_dist_size = [1, planning_horizon, self._action_dim]
        self._initial_action_mean = initial_action_mean * torch.ones(size=action_dist_size, device=device)
        self._initial_action_std = initial_action_std * torch.ones(size=action_dist_size, device=device)
        self._min_actions = min_action.to(device=device)
        self._max_actions = max_action.to(device=device)
        self._opt_steps = opt_steps
        self._planning_horizon = planning_horizon
        self._num_samples = num_samples
        self._take_best_k = take_best_k

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool,
                       obs_valid: Optional[list[torch.Tensor]]) -> tuple[torch.Tensor, dict]:
        post_state = self.model.get_next_posterior(post_state=policy_state,
                                                   observation=observation,
                                                   action=prev_action,
                                                   obs_valid=obs_valid)
        action = self.plan(state=post_state)
        return action, post_state

    def _broadcast_to_num_samples(self,
                                  x: torch.Tensor) -> torch.Tensor:
        assert self._num_samples % x.shape[0] == 0
        tile_size = int(self._num_samples // x.shape[0])
        return x.tile(tile_size, 1)

    def plan(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = 1
        action_mean = self._initial_action_mean.repeat(batch_size, 1, 1)
        action_std = self._initial_action_std.repeat(batch_size, 1, 1)

        state = {k: self._broadcast_to_num_samples(v) for k, v in state.items()}
        for i in range(self._opt_steps):
            eps = torch.randn(size=(self._num_samples, batch_size, self._planning_horizon, self._action_dim),
                              device=action_mean.device,
                              dtype=action_mean.dtype)
            action_samples = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * eps
            action_samples.clamp_(min=self._min_actions, max=self._max_actions)
            action_samples_flat = action_samples.reshape(-1, self._planning_horizon, self._action_dim)
            pred_returns_flat = \
                self.model.predict_rewards_open_loop(initial_state=state,
                                                     actions=action_samples_flat)
            pred_returns = pred_returns_flat.reshape(self._num_samples, batch_size, self._planning_horizon, 1)
            sum_pred_returns = pred_returns.sum(dim=2, keepdim=True)
            _, best_idx = torch.topk(sum_pred_returns, k=self._take_best_k, dim=0, largest=True)
            best_actions = torch.gather(action_samples, dim=0,
                                        index=best_idx.repeat(1, 1, self._planning_horizon, self._action_dim))
            action_std, action_mean = torch.std_mean(best_actions, dim=0, keepdim=False)

        return action_mean[:, 0]
