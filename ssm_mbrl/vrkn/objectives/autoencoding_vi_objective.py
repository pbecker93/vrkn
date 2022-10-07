import torch
from typing import Optional

from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from ssm_mbrl.vrkn.vrkn import VRKN
jit = torch.jit


class AEVIObjective(AbstractObjective):

    def __init__(self,
                 vrkn: VRKN,
                 decoder_loss_scales: list[float],
                 kl_scale_factor: float,
                 initial_kl_scale_factor: float,
                 free_nats: float):

        super(AEVIObjective, self).__init__(model=vrkn,
                                            decoder_loss_scales=decoder_loss_scales)

        self._vrkn = vrkn

        self._kl_scale_factor = kl_scale_factor
        self._initial_kl_scale_factor = initial_kl_scale_factor
        self._free_nats = free_nats

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  targets: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  obs_valid: Optional[list[torch.Tensor]] = None,
                                  loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                                  smoothed_states_if_avail: bool = False):

        embedded_obs = self._vrkn.encode(observations=observations, obs_valid=obs_valid)

        post_states, prior_states, transition_models = \
            self._vrkn.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                     actions=actions,
                                                     obs_valid=obs_valid)

        smoothed_states = self._vrkn.transition_model.extended_backward_pass(prior_states=prior_states,
                                                                             post_states=post_states,
                                                                             transition_models=transition_models,
                                                                             full_smoothing=True)
        smoothed_states = self._vrkn.transition_model.stack_dicts(smoothed_states)

        post_states = self._vrkn.transition_model.stack_dicts(post_states)
        prior_states = self._vrkn.transition_model.stack_dicts(prior_states)

        dec_features = self._vrkn.transition_model.get_features(state=smoothed_states)
        recon_loss, recon_lls, recon_mses = self._get_reconstruction_losses(dec_features=dec_features,
                                                                            targets=targets,
                                                                            element_wise_loss_masks=loss_masks,
                                                                            sample_wise_loss_masks=obs_valid)
        kl_term, kl_dict = self._compute_kl_term(prior_states=prior_states,
                                                 smoothed_states=smoothed_states)
        elbo = recon_loss - kl_term
        log_dict = self._build_log_dict(recon_lls=recon_lls, recon_mses=recon_mses, kl_dict=kl_dict)
        return_states = smoothed_states if (smoothed_states is not None and smoothed_states_if_avail) else post_states
        return - elbo, log_dict, return_states

    def _compute_kl_term(self,
                         prior_states: dict,
                         smoothed_states: Optional[dict]) -> tuple[torch.Tensor, dict]:

        initial_kl = self._compute_kl_filtered(lhs_mean=smoothed_states['mean'],
                                               lhs_var=smoothed_states['var'],
                                               rhs_mean=prior_states['mean'],
                                               rhs_var=prior_states['var'],
                                               time_slice=slice(0, 1))
        dyn_kl = self._compute_kl_filtered(lhs_mean=smoothed_states['cond_dyn_mean'],
                                           lhs_var=smoothed_states['cond_dyn_var'],
                                           rhs_mean=smoothed_states['dyn_mean'],
                                           rhs_var=smoothed_states['dyn_var'],
                                           time_slice=slice(1, None))
        log_dict = {'initial kl divergence': initial_kl.mean().detach().cpu().numpy()}
        initial_kl = torch.maximum(torch.zeros_like(initial_kl), initial_kl - self._free_nats).mean()
        initial_kl *= self._initial_kl_scale_factor
        log_dict['kl divergence'] = dyn_kl.mean().detach().cpu().numpy()
        dyn_kl = self._kl_scale_factor * torch.maximum(torch.zeros_like(dyn_kl), dyn_kl - self._free_nats).mean()
        total_ts = smoothed_states["cond_dyn_mean"].shape[1]
        kl_term = (1 / total_ts) * initial_kl + ((total_ts - 1) / total_ts) * dyn_kl
        return kl_term, log_dict