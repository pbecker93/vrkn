import torch
from typing import Optional, Callable

from ssm_mbrl.common.activation import DiagGaussActivation, ScaledShiftedSigmoidActivation
from ssm_mbrl.vrkn.transition.initial_state import InitialStateModel
import ssm_mbrl.common.modules as mod
import ssm_mbrl.common.bayesian_modules as mcd

nn = torch.nn
jit = torch.jit


class VRKNTM(jit.ScriptModule):

    def __init__(self,
                 obs_sizes: list[int],
                 action_dim: int,
                 lod: int,
                 mc_drop_prob: float,
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
                                     mc_drop_prob=mc_drop_prob,
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
                                mc_drop_prob: float,
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
                    getattr(nn, activation)(),
                    mcd.MCDropout(p=mc_drop_prob)
                )
        self._fake_gru = nn.GRUCell(input_size=hidden_size, hidden_size=self._lod)
        self._tm_hidden_net = torch.nn.Sequential(
                    mcd.MCDropout(p=mc_drop_prob),
                    nn.Linear(in_features=self._lod, out_features=hidden_size),
                    getattr(nn, activation)(),
                    mcd.MCDropout(p=mc_drop_prob),
                    nn.Linear(in_features=hidden_size, out_features=3 * self._lod)
                )

        self._tm_activation = ScaledShiftedSigmoidActivation(init_val=init_ev, min_val=min_ev, max_val=max_ev)
        if trans_var_sigmoid:
            self._tv_activation = ScaledShiftedSigmoidActivation(init_val=init_trans_var,
                                                                 min_val=min_trans_var,
                                                                 max_val=max_trans_var)
        else:
            self._tv_activation = DiagGaussActivation(init_var=init_trans_var, min_var=min_trans_var)

    def _build_observation_model(self,
                                 obs_sizes: list[int],
                                 init_obs_var: float,
                                 min_obs_var: float,
                                 max_obs_var: float,
                                 obs_var_sigmoid: bool,
                                 obs_output_normalization: str):
        # Latent Obs Net
        self._latent_obs_nets = nn.ModuleList()
        for os in obs_sizes:
            self._latent_obs_nets.append(
                mod.DiagonalGaussianParameterLayer(in_features=os,
                                                   distribution_dim=self._lod,
                                                   init_var=init_obs_var,
                                                   min_var=min_obs_var,
                                                   max_var=max_obs_var,
                                                   sigmoid_activation=obs_var_sigmoid,
                                                   output_normalization=obs_output_normalization))

    # interface
    @property
    def feature_size(self) -> int:
        return self._lod

    # run model parts

    @jit.script_method
    def _get_latent_obs(self, embedded_obs: list[torch.Tensor]):
        obs, obs_var = jit.annotate(list[torch.Tensor], []), jit.annotate(list[torch.Tensor], [])
        for i, lon in enumerate(self._latent_obs_nets):
            o, ov = torch.jit.annotate(tuple[torch.Tensor, torch.Tensor], lon(embedded_obs[i]))
            obs.append(o)
            obs_var.append(ov)
        return obs, obs_var

    @jit.script_method
    def _get_transition_model(self,
                              state: dict[str, torch.Tensor],
                              action: torch.Tensor) -> dict[str, torch.Tensor]:
        tm_in = torch.cat([state["mean"], action], dim=-1)
        raw_out = self._tm_hidden_net(self._fake_gru(self._tm_pre_layers(tm_in), state["mean"]))
        tm_raw, tv_raw, offset = torch.chunk(raw_out, chunks=3, dim=-1)
        tm = self._tm_activation(tm_raw)
        tv = self._tv_activation(tv_raw)
        return {"tm": tm, "tv": tv, "offset": offset}

    # Kalman Equations
    @jit.script_method
    def sample(self, mean: torch.Tensor,
               var: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(mean) * var.sqrt() + mean

    @jit.script_method
    def predict(self,
                post_state: dict[str, torch.Tensor],
                action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._predict(post_state=post_state,
                             transition_model=self._get_transition_model(state=post_state, action=action))

    def get_initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        initial_state_dict = self.initial_state_model(batch_size=batch_size)
        initial_state_dict["initial"] = torch.ones(size=(1, ),
                                                   device=initial_state_dict["mean"].device,
                                                   dtype=torch.bool)
        return initial_state_dict

    @jit.script_method
    def predict_sample(self,
                       sample: torch.Tensor,
                       transition_model: dict[str, torch.Tensor]) -> torch.Tensor:
        return transition_model["tm"] * sample + transition_model["offset"]

    def _predict(self,
                 post_state: dict[str, torch.Tensor],
                 transition_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prior_mean = transition_model["tm"] * post_state["mean"] + transition_model["offset"]
        prior_var = transition_model["tm"].square() * post_state["var"] + transition_model["tv"]

        return {"mean": prior_mean,
                "var": prior_var,
                "sample": self.sample(mean=prior_mean, var=prior_var)}

    @staticmethod
    def _kalman_update(mean: torch.Tensor,
                       var: torch.Tensor,
                       obs: torch.Tensor,
                       obs_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kalman_gain = var / (var + obs_var)
        o_m_kg = 1 - kalman_gain
        new_mean = o_m_kg * mean + kalman_gain * obs
        # Joseph Form
        new_var = o_m_kg.square() * var + kalman_gain.square() * obs_var
        return new_mean, new_var

    def _update(self,
                prior_state: dict[str, torch.Tensor],
                obs: list[torch.Tensor],
                obs_var: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        post_mean, post_var = prior_state["mean"], prior_state["var"]
        for i in range(len(obs)):
            post_mean, post_var = self._kalman_update(mean=post_mean, var=post_var, obs=obs[i], obs_var=obs_var[i])
        return {"mean": post_mean, "var": post_var, "sample": self.sample(mean=post_mean, var=post_var)}

    @jit.script_method
    def update(self,
               prior_state: dict[str, torch.Tensor],
               embedded_obs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs, obs_var = self._get_latent_obs(embedded_obs=embedded_obs)
        return self._update(prior_state=prior_state, obs=obs, obs_var=obs_var)

    def _update_if_valid(self,
                         prior_state: dict[str, torch.Tensor],
                         obs: list[torch.Tensor],
                         obs_var: list[torch.Tensor],
                         obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        post_mean, post_var = prior_state["mean"], prior_state["var"]
        for i in range(len(obs)):
            tmp_mean, tmp_var = self._kalman_update(mean=post_mean, var=post_var, obs=obs[i], obs_var=obs_var[i])
            post_mean = tmp_mean.where(obs_valid[i], post_mean)
            post_var = tmp_var.where(obs_valid[i], post_var)
        return {"mean": post_mean, "var": post_var, "sample": self.sample(mean=post_mean, var=post_var)}

    @jit.script_method
    def update_if_valid(self,
                        prior_state: dict[str, torch.Tensor],
                        embedded_obs: list[torch.Tensor],
                        obs_valid: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        obs, obs_var = self._get_latent_obs(embedded_obs=embedded_obs)
        return self._update_if_valid(prior_state=prior_state, obs=obs, obs_var=obs_var, obs_valid=obs_valid)

    @jit.script_method
    def _rts_backward(self,
                      transition_model: dict[str, torch.Tensor],
                      smoothed_state: dict[str, torch.Tensor],
                      prior_state: dict[str, torch.Tensor],
                      prev_post_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        c = prev_post_state["var"] * transition_model["tm"] / prior_state["var"]
        mean = prev_post_state["mean"] + c * (smoothed_state["mean"] - prior_state["mean"])
        var = prev_post_state["var"] + c.square() * (smoothed_state["var"] - prior_state["var"])
        return {"mean": mean, "var": var, "sample": self.sample(mean=mean, var=var)}, c

    @jit.script_method
    def rts_backward(self,
                     transition_model: dict[str, torch.Tensor],
                     smoothed_state: dict[str, torch.Tensor],
                     prior_state: dict[str, torch.Tensor],
                     prev_post_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._rts_backward(transition_model=transition_model,
                                  smoothed_state=smoothed_state,
                                  prior_state=prior_state,
                                  prev_post_state=prev_post_state)[0]

    @jit.script_method
    def extended_rts_backward(self,
                              transition_model: dict[str, torch.Tensor],
                              smoothed_state: dict[str, torch.Tensor],
                              prior_state: dict[str, torch.Tensor],
                              prev_post_state: dict[str, torch.Tensor]) \
            -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        prev_smoothed_state, c = self._rts_backward(transition_model=transition_model,
                                                    smoothed_state=smoothed_state,
                                                    prior_state=prior_state,
                                                    prev_post_state=prev_post_state)
        # q(z_t | z_{t-1}, o_{\geq t})
        cond_dyn_mat = smoothed_state["var"] * c / prev_smoothed_state["var"]
        residual = prev_smoothed_state["sample"] - prev_smoothed_state["mean"]
        cond_dyn_mean = smoothed_state["mean"] + cond_dyn_mat * residual
        cond_dyn_var = smoothed_state["var"] - cond_dyn_mat * c * smoothed_state["var"]
        return prev_smoothed_state, cond_dyn_mean, cond_dyn_var

    # Forward

    @staticmethod
    def stack_dicts(dicts: list[dict[str, torch.Tensor]], dim: int = 1) -> dict[str, torch.Tensor]:
        return {k: torch.stack([d[k] for d in dicts], dim=dim) for k in dicts[0].keys()}

    def forward_pass(self,
                     embedded_obs: list[torch.Tensor],
                     actions: torch.Tensor,
                     obs_valid: Optional[list[torch.Tensor]] = None) \
            -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        if obs_valid is None:
            return self._forward_pass(embedded_obs=embedded_obs, actions=actions)
        else:
            return self._forward_pass_with_obs_valid(embedded_obs=embedded_obs, actions=actions, obs_valid=obs_valid)

    @jit.script_method
    def _forward_pass(self, embedded_obs: list[torch.Tensor], actions: torch.Tensor) \
            -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        batch_size, seq_length = embedded_obs[0].shape[:2]
        embedded_obs = [torch.unbind(lo, 1) for lo in embedded_obs]
        actions = torch.unbind(actions, 1)

        priors = jit.annotate(list[dict[str, torch.Tensor]], [])
        posts = jit.annotate(list[dict[str, torch.Tensor]], [])
        transition_models = jit.annotate(list[dict[str, torch.Tensor]], [])

        post_state = self.initial_state_model(batch_size=batch_size)
        for i in range(seq_length):
            if i == 0:
                transition_model = {"mean": torch.ones(1, )}
                prior_state = post_state
            else:
                transition_model = self._get_transition_model(state=post_state, action=actions[i])
                prior_state = self._predict(post_state=post_state, transition_model=transition_model)
            post_state = self.update(prior_state=prior_state, embedded_obs=[eo[i] for eo in embedded_obs])
            priors.append(prior_state)
            posts.append(post_state)
            transition_models.append(transition_model)

        return posts, priors, transition_models

    @jit.script_method
    def _forward_pass_with_obs_valid(self,
                                     embedded_obs: list[torch.Tensor],
                                     actions: torch.Tensor,
                                     obs_valid: list[torch.Tensor]) \
            -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        batch_size, seq_length = embedded_obs[0].shape[:2]
        embedded_obs = [torch.unbind(lo, 1) for lo in embedded_obs]
        obs_valid = [torch.unbind(ov, 1) for ov in obs_valid]
        actions = torch.unbind(actions, 1)

        priors = jit.annotate(list[dict[str, torch.Tensor]], [])
        posts = jit.annotate(list[dict[str, torch.Tensor]], [])
        transition_models = jit.annotate(list[dict[str, torch.Tensor]], [])

        post_state = self.initial_state_model(batch_size=batch_size)
        for i in range(seq_length):
            if i == 0:
                transition_model = {"mean": torch.ones(1, )}
                prior_state = post_state
            else:
                transition_model = self._get_transition_model(state=post_state, action=actions[i])
                prior_state = self._predict(post_state=post_state, transition_model=transition_model)
            post_state = self.update_if_valid(prior_state=prior_state,
                                              embedded_obs=[eo[i] for eo in embedded_obs],
                                              obs_valid=[ov[i] for ov in obs_valid])
            priors.append(prior_state)
            posts.append(post_state)
            transition_models.append(transition_model)

        return posts, priors, transition_models

    @jit.script_method
    def backward_pass(self,
                      post_states: list[dict[str, torch.Tensor]],
                      prior_states: list[dict[str, torch.Tensor]],
                      transition_models: list[dict[str, torch.Tensor]],
                      full_smoothing: bool) -> list[dict[str, torch.Tensor]]:
        smoothed_states = jit.annotate(list[dict[str, torch.Tensor]], [{}] * len(post_states))
        smoothed_states[-1] = post_states[-1]

        for i in range(len(post_states) - 2, -1, -1):
            smoothed_states[i] = self.rts_backward(
                transition_model=transition_models[i + 1],
                smoothed_state=smoothed_states[i + 1] if full_smoothing else post_states[i + 1],
                prior_state=prior_states[i + 1],
                prev_post_state=post_states[i])
        return smoothed_states

    @jit.script_method
    def extended_backward_pass(self,
                               post_states: list[dict[str, torch.Tensor]],
                               prior_states: list[dict[str, torch.Tensor]],
                               transition_models: list[dict[str, torch.Tensor]],
                               full_smoothing: bool) -> list[dict[str, torch.Tensor]]:

        smoothed_states = jit.annotate(list[dict[str, torch.Tensor]], [{}] * len(post_states))
        smoothed_states[-1] = post_states[-1]

        for i in range(len(post_states) - 2, -1, -1):
            smoothed_states[i], cond_dyn_mean, cond_dyn_var = \
                self.extended_rts_backward\
                    (
                        transition_model=transition_models[i + 1],
                        smoothed_state=smoothed_states[i + 1] if full_smoothing else post_states[i + 1],
                        prior_state=prior_states[i + 1],
                        prev_post_state=post_states[i]
                    )

            smoothed_states[i + 1]["cond_dyn_mean"] = cond_dyn_mean
            smoothed_states[i + 1]["cond_dyn_var"] = cond_dyn_var
            smoothed_states[i + 1]["dyn_mean"] = self.predict_sample(transition_model=transition_models[i + 1],
                                                                     sample=smoothed_states[i]["sample"])
            smoothed_states[i + 1]["dyn_var"] = transition_models[i + 1]["tv"]
        fake_dyn_mean, fake_dyn_var = self.get_dummy_dist(ref_state=post_states[0])
        smoothed_states[0]["dyn_mean"] = fake_dyn_mean
        smoothed_states[0]["dyn_var"] = fake_dyn_var
        smoothed_states[0]["cond_dyn_mean"] = fake_dyn_mean
        smoothed_states[0]["cond_dyn_var"] = fake_dyn_var

        return smoothed_states

    @jit.script_method
    def get_dummy_dist(self, ref_state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        dummy_mean = torch.zeros_like(ref_state["mean"])
        dummy_var = - torch.ones_like(ref_state["var"])
        return dummy_mean, dummy_var

    @jit.script_method
    def get_features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["sample"]

    def get_next_posterior(self,
                           embedded_obs: list[torch.Tensor],
                           action: torch.Tensor,
                           post_state: dict[str, torch.Tensor],
                           obs_valid: Optional[list[torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        if "initial" in post_state.keys():
            prior_state = post_state
            prior_state.pop("initial")
        else:
            prior_state = self.predict(post_state=post_state, action=action)
        if obs_valid is None:
            return self.update(prior_state=prior_state, embedded_obs=embedded_obs)
        else:
            return self.update_if_valid(prior_state=prior_state, embedded_obs=embedded_obs, obs_valid=obs_valid)

    @jit.script_method
    def open_loop_prediction(self, initial_state: dict[str, torch.Tensor], actions: torch.Tensor) \
            -> dict[str, torch.Tensor]:
        state = initial_state
        action_list = torch.unbind(actions, 1)
        states = torch.jit.annotate(list[dict[str, torch.Tensor]], [])

        for i, action in enumerate(action_list):
            transition_model = self._get_transition_model(state=state,
                                                          action=action)
            state_mean = self.predict_sample(transition_model=transition_model,
                                             sample=state["mean"])
            dyn_mean = self.predict_sample(transition_model=transition_model,
                                           sample=state["sample"])
            state = {"mean": state_mean,
                     "var": transition_model["tv"],
                     "sample": self.sample(mean=dyn_mean, var=transition_model["tv"])}
            states.append(state)

        return self.stack_dicts(states)

    def rollout_policy(self,
                       state: dict[str, torch.Tensor],
                       policy_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                       num_steps: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        states = torch.jit.annotate(list[dict[str, torch.Tensor]], [])
        log_probs = torch.jit.annotate(list[torch.Tensor], [])
        for _ in range(num_steps):
            action, log_prob = policy_fn(self.get_features(state))
            transition_model = self._get_transition_model(state=state, action=action)
            state_mean = self.predict_sample(transition_model=transition_model,
                                             sample=state["mean"])
            dyn_mean = self.predict_sample(transition_model=transition_model,
                                           sample=state["sample"])
            state = {"mean": state_mean,
                     "sample": self.sample(mean=dyn_mean, var=transition_model["tv"])}
            states.append(state)
            log_probs.append(log_prob)

        return self.stack_dicts(states), torch.stack(log_probs, dim=1)


