import numpy as np
import torch
from typing import Optional
import ssm_mbrl.common.dense_nets as dn
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
from ssm_mbrl.mbrl.common.abstract_policy import AbstractPolicy
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor

import math

nn = torch.nn
opt = torch.optim

dists = torch.distributions


class Actor(nn.Module):

    def __init__(self,
                 in_dim: int,
                 action_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 init_std: float,
                 min_std: float,
                 mean_scale: float,
                 min_action: torch.Tensor,
                 max_action: torch.Tensor,
                 activation: str = "ReLU"):
        super(Actor, self).__init__()
        assert (min_action == -1).all() and (max_action == 1).all(), NotImplementedError

        hidden_layers, size_last_hidden = dn.build_hidden_layers(in_features=in_dim,
                                                                 layer_sizes=num_layers * [hidden_size],
                                                                 activation=activation)
        hidden_layers.append(torch.nn.Linear(in_features=size_last_hidden, out_features=2 * action_dim))
        self._actor_net = nn.Sequential(*hidden_layers)

        self._mean_scale = mean_scale
        self._min_std = min_std
        self._raw_init_std = np.log(np.exp(init_std) - 1.0)

        self.action_dim = action_dim

    def _get_base_dist(self, in_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mean, raw_std = torch.chunk(self._actor_net(in_features), chunks=2, dim=-1)

        # Taken directly from the dreamer (and dreamer v2) implementations
        mean = self._mean_scale * torch.tanh(raw_mean / self._mean_scale)
        std = nn.functional.softplus(raw_std + self._raw_init_std) + self._min_std
        return mean, std

    def forward(self, in_features: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mean, std = self._get_base_dist(in_features=in_features)
        if sample:
            return torch.tanh(mean + torch.randn_like(mean) * std)
        else:
            return torch.tanh(mean)

    def get_sampled_action_and_log_prob(self, in_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._get_base_dist(in_features=in_features)
        untransformed_sample = mean + torch.randn_like(mean) * std
        transformed_sample_log_prob = self._tanh_log_prob(untransformed_sample, mean, std)
        return torch.tanh(untransformed_sample), transformed_sample_log_prob

    @staticmethod
    @torch.jit.script
    def _tanh_log_prob(untransformed_sample, mean, std):
        base_dist_log_prob = \
            - std.log() - math.log(math.sqrt(2 * math.pi)) - 0.5 * (untransformed_sample - mean).square() / std.square()
        log_abs_det_jacobian =\
            2. * (math.log(2.) - untransformed_sample - nn.functional.softplus(-2. * untransformed_sample))
        return base_dist_log_prob.sum(dim=-1) - log_abs_det_jacobian.sum(dim=-1)


class ValueFn(nn.Module):

    def __init__(self,
                 in_dim,
                 num_layers: int,
                 hidden_size: int,
                 activation: str = "ReLU"):

        super(ValueFn, self).__init__()

        hidden_layers, size_last_hidden = dn.build_hidden_layers(in_features=in_dim,
                                                                 layer_sizes=num_layers * [hidden_size],
                                                                 activation=activation)
        hidden_layers.append(torch.nn.Linear(in_features=size_last_hidden,
                                             out_features=1))
        self._v_net = nn.Sequential(*hidden_layers)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self._v_net(in_features)


class AVPolicy(AbstractPolicy, torch.nn.Module):

    def __init__(self,
                 model: AbstractSSM,
                 actor: Actor,
                 critic: ValueFn,
                 action_dim: int,
                 obs_are_images: list[bool],
                 img_preprocessor: ImgPreprocessor,
                 device: torch.device,
                 mean_for_actor: bool):
        super(AVPolicy, self).__init__(model=model,
                                       action_dim=action_dim,
                                       obs_are_images=obs_are_images,
                                       img_preprocessor=img_preprocessor,
                                       device=device)
        self._mean_for_actor = mean_for_actor
        self.actor = actor
        self.value = critic
        self.model = model

    def _call_internal(self,
                       observation: list[torch.Tensor],
                       prev_action: torch.Tensor,
                       policy_state: dict,
                       sample: bool,
                       obs_valid: Optional[list[torch.Tensor]]) -> tuple[torch.Tensor, dict]:
        post_state = self.model.get_next_posterior(observation=observation,
                                                   action=prev_action,
                                                   post_state=policy_state,
                                                   obs_valid=obs_valid)
        features = self.model.get_features(post_state)
        action = self.actor(features)
        return action, post_state
