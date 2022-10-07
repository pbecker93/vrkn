import numpy as np
import torch
from collections import OrderedDict
from typing import Optional

from ssm_mbrl.mbrl.common.abstract_trainer import AbstractTrainer
from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from ssm_mbrl.mbrl.actor_value.av_policy import AVPolicy
from ssm_mbrl.util.freeze_parameters import FreezeParameters
nn = torch.nn
opt = torch.optim
data = torch.utils.data


class AVPolicyTrainer(AbstractTrainer):

    def __init__(self,
                 policy: AVPolicy,
                 model_objective: AbstractObjective,
                 discount: float,
                 lambda_: float,
                 imagine_horizon: int,
                 imagine_from_smoothed: bool,
                 model_learning_rate: float,
                 model_adam_eps: float,
                 model_clip_norm: float,
                 model_weight_decay: float,
                 actor_learning_rate: float,
                 actor_adam_eps: float,
                 actor_clip_norm: float,
                 actor_weight_decay: float,
                 value_learning_rate: float,
                 value_adam_eps: float,
                 value_clip_norm: float,
                 value_weight_decay: float,
                 entropy_bonus: float = 0.0,
                 eval_interval: int = 1):
        super().__init__(objective=model_objective,
                         model=policy.model,
                         eval_interval=eval_interval)

        self._policy = policy

        self._imagine_horizon = imagine_horizon
        self._imagine_from_smoothed = imagine_from_smoothed
        self._discount = discount
        self._lambda = lambda_

        self._entropy_bonus = entropy_bonus

        self._model_optimizer, self._model_clip_fn = \
            self._build_optimizer_and_clipping(params=self._objective.parameters(),
                                               learning_rate=model_learning_rate,
                                               adam_eps=model_adam_eps,
                                               clip_norm=model_clip_norm,
                                               weight_decay=model_weight_decay)
        self._actor_optimizer, self._actor_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.actor.parameters(),
                                               learning_rate=actor_learning_rate,
                                               adam_eps=actor_adam_eps,
                                               clip_norm=actor_clip_norm,
                                               weight_decay=actor_weight_decay)
        self._value_optimizer, self._value_clip_fn = \
            self._build_optimizer_and_clipping(params=self._policy.value.parameters(),
                                               learning_rate=value_learning_rate,
                                               adam_eps=value_adam_eps,
                                               clip_norm=value_clip_norm,
                                               weight_decay=value_weight_decay)

    def get_optimizer_state_dict(self) -> dict:
        return {"model": self._model_optimizer.state_dict(),
                "actor": self._actor_optimizer.state_dict(),
                "value": self._value_optimizer.state_dict()}

    def load_optimizer_state_dict(self, state_dict: dict):
        self._model_optimizer.load_state_dict(state_dict=state_dict["model"])
        self._actor_optimizer.load_state_dict(state_dict=state_dict["actor"])
        self._value_optimizer.load_state_dict(state_dict=state_dict["value"])

    @staticmethod
    def _build_optimizer_and_clipping(params,
                                      learning_rate: float,
                                      adam_eps: float,
                                      clip_norm: float,
                                      weight_decay: float) -> tuple[opt.Optimizer, callable]:
        def clip_grads_if_desired(p):
            if clip_norm > 0:
                _ = nn.utils.clip_grad_norm_(p, clip_norm)
        if weight_decay > 0.0:
            optimizer = opt.AdamW(params, lr=learning_rate, eps=adam_eps, weight_decay=weight_decay)
        else:
            optimizer = opt.Adam(params=params, lr=learning_rate, eps=adam_eps)
        return optimizer, clip_grads_if_desired

    def _train_on_batch(self, batch):
        # Model Update
        model_loss, model_obj_log_dict, post_states = \
            self._objective.compute_losses_and_states(*batch,
                                                      smoothed_states_if_avail=self._imagine_from_smoothed)
        # Actor Update
        with FreezeParameters([self._model, self._policy.value]):
            size = post_states["sample"].shape[0] * post_states["sample"].shape[1]
            initial_states = {k: v.detach().reshape(size, *v.shape[2:]) for k, v in post_states.items()}
            imagined_states, action_log_probs = \
                self._model.rollout_policy(state=initial_states,
                                           policy_fn=self._policy.actor.get_sampled_action_and_log_prob,
                                           num_steps=self._imagine_horizon)
            imagined_features = self._model.get_features(state=imagined_states)
            rewards = self._model.decoders[-1](imagined_features)[0]
            values = self._policy.value(imagined_features)

            lambda_returns = self.compute_generalized_values(rewards=rewards[:, :-1],
                                                             values=values[:, :-1],
                                                             bootstrap=values[:, -1],
                                                             discount=self._discount,
                                                             lambda_=self._lambda)
            actor_entropy = - action_log_probs.mean()
            scaled_actor_entropy = self._entropy_bonus * actor_entropy
            actor_loss = - (lambda_returns.mean() + scaled_actor_entropy)

        # Value Update
        with FreezeParameters([self._model, self._policy.actor]):
            with torch.no_grad():
                detached_imagined_features = imagined_features[:, :-1].detach()
                detached_target_values = lambda_returns.detach()
            value_loss = 0.5 * (detached_target_values - self._policy.value(detached_imagined_features)).square().mean()

        self._model_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        self._value_optimizer.zero_grad()

        model_loss.backward()
        actor_loss.backward()
        value_loss.backward()

        self._model_clip_fn(self._model.parameters())
        self._actor_clip_fn(self._policy.actor.parameters())
        self._value_clip_fn(self._policy.value.parameters())

        self._model_optimizer.step()
        self._actor_optimizer.step()
        self._value_optimizer.step()

        log_dict = OrderedDict({"model/neg_elbo": model_loss.detach_().cpu().numpy()},
                               **{"model/{}".format(k): v for k, v in model_obj_log_dict.items()})
        log_dict["actor/loss"] = actor_loss.detach_().cpu().numpy()
        log_dict["value/loss"] = value_loss.detach_().cpu().numpy()
        log_dict["actor/entropy"] = actor_entropy.detach_().cpu().numpy()

        return log_dict

    @staticmethod
    def compute_generalized_values(rewards: torch.Tensor,
                                   values: torch.Tensor,
                                   bootstrap: torch.Tensor,
                                   discount: float = 0.99,
                                   lambda_: float = 0.95) -> torch.Tensor:
        next_values = torch.cat([values[:, 1:], bootstrap[:, None]], 1)
        deltas = rewards + discount * next_values * (1 - lambda_)
        last = bootstrap
        returns = torch.ones_like(rewards)
        for t in reversed(range(rewards.shape[1])):
            returns[:, t] = last = deltas[:, t] + (discount * lambda_ * last)
        return returns