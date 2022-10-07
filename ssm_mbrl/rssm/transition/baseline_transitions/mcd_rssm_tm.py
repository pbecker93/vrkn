import torch
from typing import Optional
import ssm_mbrl.common.bayesian_modules as mcd
import ssm_mbrl.common.modules as mod

from ssm_mbrl.rssm.transition.r_rssm_tm import RRSSMTM
from ssm_mbrl.rssm.transition.baseline_transitions.s_rssm_tm import SRSSMTM

nn = torch.nn
jit = torch.jit


class McDropoutRRSSMTM(RRSSMTM):

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 mc_drop_prob: float,
                 build_with_obs_valid: bool,
                 state_part_for_update: str = "d",  # either "d(eterministic)", "s(tochastic)" or "b(oth)"
                 activation: str = "ReLU"):
        self._mc_drop_prob = mc_drop_prob
        super(McDropoutRRSSMTM, self).__init__(obs_sizes=obs_sizes,
                                               state_dim=state_dim,
                                               action_dim=action_dim,
                                               rec_state_dim=rec_state_dim,
                                               num_layers=num_layers,
                                               hidden_size=hidden_size,
                                               min_std=min_std,
                                               build_with_obs_valid=build_with_obs_valid,
                                               state_part_for_update=state_part_for_update,
                                               activation=activation)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str,
                       min_std: float):
        pre_layers, pre_hidden_out_size = \
            mcd.build_hidden_layers_with_mc_dropout(in_features=self._state_dim + action_dim,
                                                    drop_prob=self._mc_drop_prob,
                                                    layer_sizes=[hidden_size] * num_layers,
                                                    hidden_act=activation)
        self._pred_pre_hidden_layers = nn.Sequential(*pre_layers)
        self._pred_tm_cell = nn.GRUCell(input_size=pre_hidden_out_size,
                                        hidden_size=self._rec_state_dim)

        post_layers, post_out_size = mcd.build_hidden_layers_with_mc_dropout(in_features=self._rec_state_dim,
                                                                             drop_prob=self._mc_drop_prob,
                                                                             layer_sizes=[hidden_size] * num_layers,
                                                                             hidden_act=activation)
        post_layers = torch.nn.ModuleList([mcd.MCDropout(p=self._mc_drop_prob)] + list(post_layers))
        post_layers.append(mod.DiagonalGaussianParameterLayer(in_features=post_out_size,
                                                              distribution_dim=self._state_dim,
                                                              init_var=1.0,
                                                              min_var=min_std))
        self._pred_post_layers = nn.Sequential(*post_layers)


class FullMCDropoutRRSSMTM(McDropoutRRSSMTM):

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 rec_state_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 mc_drop_prob: float,
                 build_with_obs_valid: bool,
                 state_part_for_update: str = "d",  # either "d(eterministic)", "s(tochastic)" or "b(oth)"
                 activation: str = "ReLU"):
        assert state_part_for_update in ["d", "det", "deterministic"], NotImplementedError
        super(FullMCDropoutRRSSMTM, self).__init__(obs_sizes=obs_sizes,
                                                   state_dim=state_dim,
                                                   action_dim=action_dim,
                                                   rec_state_dim=rec_state_dim,
                                                   num_layers=num_layers,
                                                   hidden_size=hidden_size,
                                                   min_std=min_std,
                                                   mc_drop_prob=mc_drop_prob,
                                                   build_with_obs_valid=build_with_obs_valid,
                                                   state_part_for_update=state_part_for_update,
                                                   activation=activation)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str,
                       min_std: float):
        super(FullMCDropoutRRSSMTM, self)._build_predict(action_dim=action_dim,
                                                         num_layers=num_layers,
                                                         hidden_size=hidden_size,
                                                         activation=activation,
                                                         min_std=min_std)
        self._post_gru_mcd = mcd.MCDropout(p=self._mc_drop_prob)

    def _build_update(self,
                      obs_sizes: list[int],
                      num_layers: int,
                      hidden_size: int,
                      activation: str,
                      min_std: float):

        inpt_size = sum(obs_sizes) + (len(obs_sizes) if self._built_with_obs_valid else 0) + self._prior_input_size
        self._updt_dist_layer = nn.Sequential(
            nn.Linear(in_features=inpt_size, out_features=hidden_size),
            getattr(nn, activation)(),
            mcd.MCDropout(p=self._mc_drop_prob),
            mod.SimpleGaussianParameterLayer(in_features=hidden_size,
                                             distribution_dim=self._state_dim,
                                             min_std_or_var=min_std))

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        trans_input = post_state["sample"] if action is None else torch.cat([post_state["sample"], action], dim=-1)
        cell_state = self._pred_tm_cell(self._pred_pre_hidden_layers(trans_input),
                                        post_state["gru_cell_state"])
        dropped_out_cell_state = self._post_gru_mcd(cell_state)
        m, s = self._pred_post_layers(dropped_out_cell_state)
        return {"mean": m,
                "std": s,
                "sample": self.sample_gauss(m, s),
                "gru_cell_state": cell_state,
                "dropped_out_cell_state": dropped_out_cell_state}

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pm, ps = self._updt_dist_layer(torch.cat(obs + [prior_state["dropped_out_cell_state"]], dim=-1))
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
        pm, ps = self._updt_dist_layer(torch.cat(obs + obs_valid + [prior_state["dropped_out_cell_state"]], dim=-1))
        return {"mean": pm,
                "std": ps,
                "sample": self.sample_gauss(pm, ps),
                "gru_cell_state": prior_state["gru_cell_state"]}


class McDropoutSRSSMTM(SRSSMTM):

    def __init__(self,
                 obs_sizes: list[int],
                 state_dim: int,
                 action_dim: int,
                 num_layers: int,
                 hidden_size: int,
                 min_std: float,
                 mc_drop_prob: float,
                 build_with_obs_valid: bool,
                 activation: str = "ReLU"):
        self._mc_drop_prob = mc_drop_prob
        super(McDropoutSRSSMTM, self).__init__(obs_sizes=obs_sizes,
                                               state_dim=state_dim,
                                               action_dim=action_dim,
                                               num_layers=num_layers,
                                               hidden_size=hidden_size,
                                               min_std=min_std,
                                               build_with_obs_valid=build_with_obs_valid,
                                               activation=activation)

    def _build_predict(self,
                       action_dim: int,
                       num_layers: int,
                       hidden_size: int,
                       activation: str,
                       min_std: float):
        pred_hidden_layers, last_hidden_size = \
            mcd.build_hidden_layers_with_mc_dropout(in_features=self._state_dim + action_dim,
                                                    drop_prob=self._mc_drop_prob,
                                                    layer_sizes=[hidden_size] * num_layers,
                                                    hidden_act=activation)
        pred_hidden_layers.append(mod.DiagonalGaussianParameterLayer(in_features=last_hidden_size,
                                                                     distribution_dim=self._state_dim,
                                                                     init_var=1.0,
                                                                     min_var=min_std))
        self._pred_net = nn.Sequential(*pred_hidden_layers)