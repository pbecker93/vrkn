import torch
from typing import Optional

from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from ssm_mbrl.rssm.rssm import RSSM


class AEVIObjective(AbstractObjective):

    def __init__(self,
                 rssm: RSSM,
                 decoder_loss_scales: list[float] = [1.0],
                 kl_scale_factor: float = 1.0,
                 free_nats: float = 0.0):

        super(AEVIObjective, self).__init__(model=rssm,
                                            decoder_loss_scales=decoder_loss_scales)
        self._rssm = rssm
        self._kl_scale_factor = kl_scale_factor
        self._free_nats = free_nats

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  targets: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  obs_valid: Optional[list[Optional[torch.Tensor]]] = None,
                                  loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                                  smoothed_states_if_avail: bool = False):
        assert not smoothed_states_if_avail

        embedded_obs = self._rssm.encode(observations=observations, obs_valid=obs_valid)
        post_states, prior_states = self._rssm.transition_model.forward_pass(embedded_obs=embedded_obs,
                                                                             actions=actions,
                                                                             obs_valid=obs_valid)
        dec_features = self._rssm.transition_model.get_features(post_states)
        recon_loss, recon_lls, recon_mses = self._get_reconstruction_losses(dec_features=dec_features,
                                                                            targets=targets,
                                                                            element_wise_loss_masks=loss_masks,
                                                                            sample_wise_loss_masks=obs_valid)

        kl_term, kl_dict = self._compute_kl_term(lhs_state=post_states,
                                                 rhs_state=prior_states,
                                                 scale_factor=self._kl_scale_factor,
                                                 free_nats=self._free_nats)
        elbo = recon_loss - kl_term
        return - elbo, self._build_log_dict(recon_lls=recon_lls, recon_mses=recon_mses, kl_dict=kl_dict), post_states

    def _compute_kl_term(self,
                         lhs_state: dict,
                         rhs_state: dict,
                         scale_factor: float,
                         free_nats: float) -> tuple[torch.Tensor, dict]:
        lhs_var, rhs_var = lhs_state["std"].square(), rhs_state["std"].square()
        kl = self._compute_kl(lhs_mean=lhs_state["mean"],
                              lhs_var=lhs_var,
                              rhs_mean=rhs_state["mean"],
                              rhs_var=rhs_var)
        log_dict = {"kl divergence": kl.mean().detach().cpu().numpy()}
        return scale_factor * torch.maximum(torch.zeros_like(kl), kl - free_nats).mean(), log_dict

