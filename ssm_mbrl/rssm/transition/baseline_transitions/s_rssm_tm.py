import torch
from typing import Optional
import ssm_mbrl.common.modules as mod
import ssm_mbrl.common.dense_nets as dn
from ssm_mbrl.rssm.transition.abstract_tm import AbstractRSSMTM

nn = torch.nn
jit = torch.jit


class SRSSMTM(AbstractRSSMTM):

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 build_with_obs_valid: bool,
                 activation: str = "ReLU"):
        super(SRSSMTM, self).__init__(action_dim=action_dim, build_with_obs_valid=build_with_obs_valid)
        self._state_dim = state_dim

        self._build_predict(action_dim=action_dim,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            activation=activation,
                            min_std=min_std)

        self._build_update(obs_sizes=obs_sizes,
                           num_layers=num_layers,
                           hidden_size=hidden_size,
                           activation=activation,
                           min_std=min_std)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str,
                       min_std: float):
        pred_hidden_layers, last_hidden_size = dn.build_hidden_layers(in_features=self._state_dim + action_dim,
                                                                      layer_sizes=[hidden_size] * num_layers,
                                                                      activation=activation)
        pred_hidden_layers.append(mod.DiagonalGaussianParameterLayer(in_features=last_hidden_size,
                                                                     distribution_dim=self._state_dim,
                                                                     init_var=1.0,
                                                                     min_var=min_std))
        self._pred_net = nn.Sequential(*pred_hidden_layers)

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      hidden_size: int,
                      activation: str,
                      min_std: float):
        in_features = 2 * self._state_dim + sum(obs_sizes) + (len(obs_sizes) if self._built_with_obs_valid else 0)
        # Build Update
        updt_hidden_layers, last_hidden_size = dn.build_hidden_layers(in_features=in_features,
                                                                      layer_sizes=[hidden_size] * num_layers,
                                                                      activation=activation)
        updt_hidden_layers.append(mod.DiagonalGaussianParameterLayer(in_features=last_hidden_size,
                                                                     distribution_dim=self._state_dim,
                                                                     init_var=1.0,
                                                                     min_var=min_std))
        self._updt_net = nn.Sequential(*updt_hidden_layers)

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        post_mean, post_std = self._updt_net(torch.cat(obs + [prior_state["mean"], prior_state["std"]], dim=-1))
        return {"mean": post_mean,
                "std": post_std,
                "sample": self.sample_gauss(mean=post_mean, std=post_std)}

    @jit.script_method
    def update_with_obs_valid(self,
                              prior_state: dict[str, torch.Tensor],
                              obs: list[torch.Tensor],
                              obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = [o.where(ov, torch.ones_like(o) * self._default_value) for o, ov in zip(obs, obs_valid)]
        obs_valid = [o.float() for o in obs_valid]
        joint_input = torch.cat(obs + obs_valid + [prior_state["mean"], prior_state["std"]], dim=-1)
        post_mean, post_std = self._updt_net(joint_input)
        return {"mean": post_mean,
                "std": post_std,
                "sample": self.sample_gauss(mean=post_mean, std=post_std)}

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        trans_input = post_state["sample"] if action is None else torch.cat([post_state["sample"], action], dim=-1)
        prior_mean, prior_std = self._pred_net(trans_input)
        return {"mean": prior_mean,
                "std": prior_std,
                "sample": self.sample_gauss(mean=prior_mean, std=prior_std)}

    @jit.script_method
    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_net[0].weight
        return {"mean": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "std": torch.ones(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "sample": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype)}

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def feature_size(self):
        return self._state_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["sample"]


