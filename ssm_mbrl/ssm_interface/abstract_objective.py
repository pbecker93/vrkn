import time

import torch
from typing import Optional
import numpy as np
from collections import OrderedDict

from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM

dists = torch.distributions
F = torch.nn.functional
nn = torch.nn


class AbstractObjective(nn.Module):
    
    LOG_SQRT_2_PI = np.log(np.sqrt(2 * np.pi))

    def __init__(self,
                 model: AbstractSSM,
                 decoder_loss_scales: list[float]):
        super(AbstractObjective, self).__init__()
        self._model = model
        self._decoder_loss_scales = decoder_loss_scales

    def compute_losses(self,
                       observations: list[torch.Tensor],
                       targets: list[torch.Tensor],
                       actions: Optional[torch.Tensor] = None,
                       obs_valid: Optional[list[Optional[torch.Tensor]]] = None,
                       loss_masks: Optional[list[Optional[torch.Tensor]]] = None) -> tuple[torch.Tensor, dict]:
        return self.compute_losses_and_states(observations=observations,
                                              targets=targets,
                                              actions=actions,
                                              obs_valid=obs_valid,
                                              loss_masks=loss_masks)[:2]

    def compute_losses_and_states(self,
                                  observations: list[torch.Tensor],
                                  targets: list[torch.Tensor],
                                  actions: torch.Tensor,
                                  obs_valid: Optional[list[Optional[torch.Tensor]]] = None,
                                  loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                                  smoothed_states_if_avail: bool = False) -> tuple[torch.Tensor, dict, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def mse(target: torch.Tensor,
            predicted: torch.Tensor,
            element_wise_mask: Optional[torch.Tensor] = None,
            sample_wise_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if element_wise_mask is not None:
            assert len(target.shape) == 5
            element_wise_mse = (target - predicted).square() * element_wise_mask.unsqueeze(2)
            num_valid_pixels = torch.sum(element_wise_mask, dim=(-2, -1)) * element_wise_mse.shape[2]
            sample_wise_mse = element_wise_mse.sum(dim=(2, 3, 4)) / num_valid_pixels
        else:
            red_axis = tuple(- (i + 1) for i in range(len(target.shape) - 2))
            sample_wise_mse = (target - predicted).square().mean(red_axis)

        if sample_wise_mask is not None:
            sample_wise_mask = sample_wise_mask.reshape(sample_wise_mse.shape)
            return (sample_wise_mse * sample_wise_mask).sum() / torch.count_nonzero(sample_wise_mask)
        else:
            return sample_wise_mse.mean()

    def _get_reconstruction_losses(self,
                                   dec_features: torch.Tensor,
                                   targets: list[torch.Tensor],
                                   element_wise_loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                                   sample_wise_loss_masks: Optional[list[Optional[torch.Tensor]]] = None):
        assert len(targets) == len(self._decoder_loss_scales)
        reconstruction_lls = []
        reconstruction_mses = []
        for i, dec in enumerate(self._model.decoders):
            e_mask = None if element_wise_loss_masks is None else element_wise_loss_masks[i]
            if sample_wise_loss_masks is not None and i < len(self._model.decoders) - 1:
                s_mask = sample_wise_loss_masks[i]
            else:
                s_mask = None
            pred_mean, pred_std = dec(dec_features, mask=s_mask)
            reconstruction_lls.append(self.gaussian_ll(target=targets[i],
                                                       predicted_mean=pred_mean,
                                                       predicted_std=pred_std,
                                                       element_wise_mask=e_mask,
                                                       sample_wise_mask=s_mask))
            with torch.no_grad():
                reconstruction_mses.append(self.mse(target=targets[i],
                                                    predicted=pred_mean,
                                                    element_wise_mask=e_mask,
                                                    sample_wise_mask=s_mask))
        recon_ll = torch.stack([s * l for s, l in zip(self._decoder_loss_scales, reconstruction_lls)], dim=0).sum(dim=0)
        return recon_ll, reconstruction_lls, reconstruction_mses

    @staticmethod
    def _build_log_dict(recon_lls: list[torch.Tensor],
                        recon_mses: list[torch.Tensor],
                        kl_dict: dict[str, float]) -> dict[str, float]:
        log_dict = \
            OrderedDict({"decoder_{}_ll".format(i): ll.detach().cpu().numpy() for i, ll in enumerate(recon_lls)})
        mse_dict = \
            OrderedDict({"decoder_{}_mse".format(i): mse.detach().cpu().numpy() for i, mse in enumerate(recon_mses)})
        log_dict |= kl_dict | mse_dict
        return log_dict

    @staticmethod
    def gaussian_ll(target: torch.Tensor,
                    predicted_mean: torch.Tensor,
                    predicted_std: torch.Tensor,
                    element_wise_mask: Optional[torch.Tensor] = None,
                    sample_wise_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_term = - predicted_std.log() - AbstractObjective.LOG_SQRT_2_PI
        exp_term = - 0.5 * (target - predicted_mean).square() / predicted_std.square()

        if element_wise_mask is not None:
            assert len(target.shape) == 5
            element_wise_nll = (norm_term + exp_term) * element_wise_mask.unsqueeze(2)
            num_valid_pixels = torch.sum(element_wise_mask, dim=(-2, -1)) * element_wise_nll.shape[2]
            num_pixels = target.shape[2] * target.shape[3] * target.shape[4]
            sample_wise_nll = (element_wise_nll.sum(dim=(2, 3, 4)) / num_valid_pixels) * num_pixels
            return sample_wise_nll.mean()
        else:
            sample_wise_nll = (norm_term + exp_term).sum(tuple(- (i + 1) for i in range(len(target.shape) - 2)))

        if sample_wise_mask is not None:
            sample_wise_mask = sample_wise_mask.reshape(sample_wise_nll.shape)
            return (sample_wise_nll * sample_wise_mask).sum() / torch.count_nonzero(sample_wise_mask)
        else:
            return sample_wise_nll.mean()

    @staticmethod
    def _compute_kl(lhs_mean: torch.Tensor,
                    lhs_var: torch.Tensor,
                    rhs_mean: torch.Tensor,
                    rhs_var: torch.Tensor) -> torch.Tensor:
        log_det_term = rhs_var.log().sum(dim=-1) - lhs_var.log().sum(dim=-1)
        trace_term = (lhs_var / rhs_var).sum(dim=-1)
        mahal_term = ((rhs_mean - lhs_mean).square() / rhs_var).sum(dim=-1)
        return 0.5 * (log_det_term + trace_term - lhs_mean.shape[-1] + mahal_term)#

    @staticmethod
    def _compute_kl_filtered(lhs_mean: torch.Tensor,
                             lhs_var: torch.Tensor,
                             rhs_mean: torch.Tensor,
                             rhs_var: torch.Tensor,
                             time_slice: slice) -> torch.Tensor:
        return AbstractObjective._compute_kl(lhs_mean=lhs_mean[:, time_slice],
                                             lhs_var=lhs_var[:, time_slice],
                                             rhs_mean=rhs_mean[:, time_slice],
                                             rhs_var=rhs_var[:, time_slice])
