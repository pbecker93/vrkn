import torch
from typing import Optional
import ssm_mbrl.common.modules as mod
import ssm_mbrl.common.dense_nets as dn
from ssm_mbrl.rssm.transition.abstract_tm import AbstractRSSMTM

nn = torch.nn
jit = torch.jit


class RRSSMTM(AbstractRSSMTM):
    """Extended Version of the RSSM used in PlaNet and Dreamer (v1), to recover original model use
       build_with_obs_valid=False and state_part_for_update="d"
    """

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 build_with_obs_valid: bool,
                 state_part_for_update: str = "d",  # either "d(eterministic)", "s(tochastic)" or "b(oth)"
                 activation: str = "ReLU"):
        super(RRSSMTM, self).__init__(action_dim=action_dim, build_with_obs_valid=build_with_obs_valid)
        self._state_dim = state_dim
        self._rec_state_dim = rec_state_dim
        self._state_part_for_update = state_part_for_update
        self._min_std = min_std

        if self._state_part_for_update.lower() in ["e", "estem", "estimate"]:
            self._prior_input_size = 2 * self._state_dim
        elif self._state_part_for_update.lower() in ["s", "stoch", "sample"]:
            self._prior_input_size = self._state_dim
        elif self._state_part_for_update.lower() in ["d", "det", "deterministic"]:
            self._prior_input_size = self._rec_state_dim
        elif self._state_part_for_update.lower() in ["b", "both"]:
            self._prior_input_size = 2 * self._state_dim + self._rec_state_dim
        else:
            raise AssertionError

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
        pre_layers, pre_hidden_out_size = dn.build_hidden_layers(in_features=self._state_dim + action_dim,
                                                                 layer_sizes=[hidden_size] * num_layers,
                                                                 activation=activation)
        self._pred_pre_hidden_layers = nn.Sequential(*pre_layers)
        self._pred_tm_cell = nn.GRUCell(input_size=pre_hidden_out_size,
                                        hidden_size=self._rec_state_dim)

        post_layers, post_hidden_out_size = dn.build_hidden_layers(in_features=self._rec_state_dim,
                                                                   layer_sizes=[hidden_size] * num_layers,
                                                                   activation=activation)
        post_layers.append(mod.SimpleGaussianParameterLayer(in_features=post_hidden_out_size,
                                                            distribution_dim=self._state_dim,
                                                            min_std_or_var=min_std))
        self._pred_post_layers = nn.Sequential(*post_layers)

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      hidden_size: int,
                      activation: str,
                      min_std: float):

        inpt_size = sum(obs_sizes) + (len(obs_sizes) if self._built_with_obs_valid else 0) + self._prior_input_size
        hidden_layers, hidden_out_size = dn.build_hidden_layers(in_features=inpt_size,
                                                                layer_sizes=[hidden_size] * num_layers,
                                                                activation=activation)
        hidden_layers.append(mod.SimpleGaussianParameterLayer(in_features=hidden_out_size,
                                                              distribution_dim=self._state_dim,
                                                              min_std_or_var=min_std))
        self._updt_dist_layer = nn.Sequential(*hidden_layers)

    @jit.script_method
    def _get_prior_input(self, prior_state: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        if self._state_part_for_update.lower() in ["e", "estem", "estimate"]:
            return [prior_state["mean"], prior_state["std"]]
        elif self._state_part_for_update.lower() in ["s", "stoch", "sample"]:
            return [prior_state["sample"]]
        elif self._state_part_for_update.lower() in ["d", "det", "deterministic"]:
            return [prior_state["gru_cell_state"]]
        elif self._state_part_for_update.lower() in ["b", "both"]:
            return [prior_state["mean"], prior_state["std"], prior_state["gru_cell_state"]]
        else:
            raise AssertionError

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pm, ps = self._updt_dist_layer(torch.cat(obs + self._get_prior_input(prior_state=prior_state), dim=-1))
        return {"mean": pm,
                "std": ps,
                "sample": self.sample_gauss(pm, ps),
                "gru_cell_state": prior_state["gru_cell_state"]}

    @jit.script_method
    def update_with_obs_valid(self,
                              prior_state: dict[str, torch.Tensor],
                              obs: list[torch.Tensor],
                              obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = [o.where(ov, torch.ones_like(o) * self._default_value) for o, ov in zip(obs, obs_valid)]
        obs_valid = [ov.float() for ov in obs_valid]
        prior_input = self._get_prior_input(prior_state=prior_state)
        pm, ps = self._updt_dist_layer(torch.cat(obs + obs_valid + prior_input, dim=-1))
        return {"mean": pm,
                "std": ps,
                "sample": self.sample_gauss(pm, ps),
                "gru_cell_state": prior_state["gru_cell_state"]}

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        trans_input = post_state["sample"] if action is None else torch.cat([post_state["sample"], action], dim=-1)
        cell_state = self._pred_tm_cell(self._pred_pre_hidden_layers(trans_input),
                                        post_state["gru_cell_state"])
        m, s = self._pred_post_layers(cell_state)
        return {"mean": m,
                "std": s,
                "sample": self.sample_gauss(m, s),
                "gru_cell_state": cell_state}

    @jit.script_method
    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        p = self._pred_tm_cell.weight_ih
        return {"mean": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                # this should never be used in the rssm implementation
                "std": - torch.ones(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "sample": torch.zeros(size=[batch_size, self._state_dim], device=p.device, dtype=p.dtype),
                "gru_cell_state": torch.zeros(size=[batch_size, self._rec_state_dim],
                                              device=p.device,
                                              dtype=p.dtype)}

    @property
    def feature_size(self):
        return self._state_dim + self._rec_state_dim

    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["sample"], state["gru_cell_state"]], dim=-1)

    @property
    def latent_distribution(self) -> str:
        return "gaussian"
