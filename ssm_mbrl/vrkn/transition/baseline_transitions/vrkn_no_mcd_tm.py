import torch

from ssm_mbrl.vrkn.transition.vrkn_tm import VRKNTM
from ssm_mbrl.common.activation import DiagGaussActivation, ScaledShiftedSigmoidActivation
from ssm_mbrl.vrkn.transition.initial_state import InitialStateModel
nn = torch.nn


class VRKNNoMCDTM(VRKNTM):

    def __init__(self,
                 obs_sizes: list[int],
                 action_dim: int,
                 lod: int,
                 init_obs_var: float,
                 min_obs_var: float,
                 max_obs_var: float,
                 obs_var_sigmoid: bool,
                 obs_output_normalization: str,
                 hidden_size: int,
                 activation: str,
                 init_ev: float,
                 min_ev: float,
                 max_ev: float,
                 init_trans_var: float,
                 min_trans_var: float,
                 max_trans_var: float,
                 trans_var_sigmoid: bool,
                 learn_initial_state_var: bool,
                 initial_state_var: float,
                 min_initial_state_var: float):
        super(VRKNTM, self).__init__()

        self._lod = lod

        # Transition Model
        self._build_transition_model(action_dim=action_dim,
                                     hidden_size=hidden_size,
                                     activation=activation,
                                     init_ev=init_ev,
                                     min_ev=min_ev,
                                     max_ev=max_ev,
                                     trans_var_sigmoid=trans_var_sigmoid,
                                     init_trans_var=init_trans_var,
                                     min_trans_var=min_trans_var,
                                     max_trans_var=max_trans_var)

        self.initial_state_model = InitialStateModel(latent_state_dim=self._lod,
                                                     learn_var=learn_initial_state_var,
                                                     init_var=initial_state_var,
                                                     min_var=min_initial_state_var)

        self._build_observation_model(obs_sizes=obs_sizes,
                                      init_obs_var=init_obs_var,
                                      min_obs_var=min_obs_var,
                                      max_obs_var=max_obs_var,
                                      obs_var_sigmoid=obs_var_sigmoid,
                                      obs_output_normalization=obs_output_normalization)

    def _build_transition_model(self,
                                action_dim: int,
                                hidden_size: int,
                                activation: str,
                                init_ev: float,
                                min_ev: float,
                                max_ev: float,
                                trans_var_sigmoid: bool,
                                init_trans_var: float,
                                min_trans_var: float,
                                max_trans_var: float):
        in_dim = self._lod + action_dim
        self._tm_pre_layers = torch.nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=hidden_size),
                    getattr(nn, activation)()
                )
        self._fake_gru = nn.GRUCell(input_size=hidden_size, hidden_size=self._lod)
        self._tm_hidden_net = torch.nn.Sequential(
                    nn.Linear(in_features=self._lod, out_features=hidden_size),
                    getattr(nn, activation)(),
                    nn.Linear(in_features=hidden_size, out_features=3 * self._lod)
                )

        self._tm_activation = ScaledShiftedSigmoidActivation(init_val=init_ev, min_val=min_ev, max_val=max_ev)
        if trans_var_sigmoid:
            self._tv_activation = ScaledShiftedSigmoidActivation(init_val=init_trans_var,
                                                                 min_val=min_trans_var,
                                                                 max_val=max_trans_var)
        else:
            self._tv_activation = DiagGaussActivation(init_var=init_trans_var, min_var=min_trans_var)