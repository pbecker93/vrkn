import torch
from typing import Optional, Callable
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM

nn = torch.nn


class VRKN(AbstractSSM):

    def __init__(self,
                 encoders: nn.ModuleList,
                 transition_model,
                 decoders: nn.ModuleList):
        super().__init__()

        # Mandatory
        self.encoders = encoders
        self.transition_model = transition_model
        self.decoders = decoders

    def encode(self,
               observations: list[torch.Tensor],
               obs_valid: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        embedded_obs = []
        for i, enc in enumerate(self.encoders):
            embedded_obs.append(enc(observations[i],
                                    mask=None if obs_valid is None else obs_valid[i]))
        return embedded_obs

    @property
    def feature_size(self) -> int:
        return self.transition_model.feature_size

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.transition_model.get_features(state=state)

    def get_initial_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self.transition_model.get_initial(batch_size=batch_size)

    def get_next_posterior(self,
                           observation: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict,
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict:
        u_obs = [o.unsqueeze_(dim=1) for o in observation]
        embedded_obs = [x.squeeze(dim=1) for x in self.encode(u_obs, obs_valid=obs_valid)]

        return self.transition_model.get_next_posterior(embedded_obs=embedded_obs,
                                                        action=action,
                                                        post_state=post_state,
                                                        obs_valid=obs_valid)

    def predict_rewards_open_loop(self,
                                  initial_state: dict,
                                  actions: torch.Tensor):
        states = self.transition_model.open_loop_prediction(initial_state=initial_state,
                                                            actions=actions)
        return self.decoders[-1](self.transition_model.get_features(state=states))[0]

    def rollout_policy(self,
                       state: dict,
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict, torch.Tensor]:
        return self.transition_model.rollout_policy(state=state,
                                                    policy_fn=policy_fn,
                                                    num_steps=num_steps)
