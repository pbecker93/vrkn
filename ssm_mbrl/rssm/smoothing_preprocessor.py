import torch
from typing import Optional

import ssm_mbrl.common.dense_nets as dn

nn = torch.nn
dists = torch.distributions
jit = torch.jit


class SmoothPreprocessor(jit.ScriptModule):

    def __init__(self,
                 obs_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 activation: str,
                 build_with_obs_vaild: bool):
        super(SmoothPreprocessor, self).__init__()
        self._hidden_size = hidden_size
        self._default_value = 0.
        in_features = obs_dim + (1 if build_with_obs_vaild else 0)
        self._pre_hidden_layers = nn.Sequential(*dn.build_hidden_layers(in_features=in_features,
                                                                        layer_sizes=[hidden_size] * num_layers,
                                                                        activation=activation)[0])
        self._build_with_obs_valid = build_with_obs_vaild
        self._smooth_cell = nn.GRUCell(input_size=hidden_size,
                                       hidden_size=hidden_size)

        self._post_hidden_layers = nn.Sequential(*dn.build_hidden_layers(in_features=hidden_size,
                                                                         layer_sizes=[hidden_size] * num_layers,
                                                                         activation=activation)[0])

    def forward(self, observations: torch.Tensor, obs_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self._build_with_obs_valid == (obs_valid is not None)
        if obs_valid is None:
            return self._forward(observations=observations)
        else:
            return self._forward_with_obs_valid(observations=observations, obs_valid=obs_valid)

    @torch.jit.script_method
    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = observations.shape[:2]
        dev = observations.device
        smoothed_observations = torch.zeros(size=[batch_size, seq_length, self._hidden_size], device=dev)
        cell_state = torch.zeros(size=[batch_size, self._hidden_size], device=dev)
        for i in range(seq_length - 1, -1, -1):
            obs = observations[:, i]
            hidden = self._pre_hidden_layers(obs)
            cell_state = self._smooth_cell(hidden, cell_state)
            smoothed_observations[:, i] = self._post_hidden_layers(cell_state)
        return smoothed_observations

    @torch.jit.script_method
    def _forward_with_obs_valid(self, observations: torch.Tensor, obs_valid: torch.Tensor) -> torch.Tensor:
        obs_valid = obs_valid.reshape((observations.shape[0], observations.shape[1], 1))
        batch_size, seq_length = observations.shape[:2]
        dev = observations.device
        smoothed_observations = torch.zeros(size=[batch_size, seq_length, self._hidden_size], device=dev)
        cell_state = torch.zeros(size=[batch_size, self._hidden_size], device=dev)
        for i in range(seq_length - 1, -1, -1):
            layer_inpt = observations[:, i].where(obs_valid[:, i],
                                                  torch.ones_like(observations[:, i]) * self._default_value)
            layer_inpt = torch.cat([layer_inpt, obs_valid[:, i].float()], dim=-1)
            hidden = self._pre_hidden_layers(layer_inpt)
            cell_state = self._smooth_cell(hidden, cell_state)
            smoothed_observations[:, i] = self._post_hidden_layers(cell_state)
        return smoothed_observations
