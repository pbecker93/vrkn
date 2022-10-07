import torch
from collections import OrderedDict

from ssm_mbrl.mbrl.common.abstract_trainer import AbstractTrainer
from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM
nn = torch.nn
opt = torch.optim
data = torch.utils.data


class ModelTrainer(AbstractTrainer):

    def __init__(self,
                 model: AbstractSSM,
                 objective: AbstractObjective,
                 learning_rate: float,
                 adam_epsilon: float = 1e-8,
                 clip_grad_by_norm: bool = True,
                 clip_norm: float = 1000.0,
                 eval_interval: int = 1):
        super().__init__(objective=objective,
                         model=model,
                         eval_interval=eval_interval)

        self._clip_grad_by_norm = clip_grad_by_norm
        self._clip_norm = clip_norm

        self._optimizer = opt.Adam(params=self._model.parameters(),
                                   lr=learning_rate,
                                   eps=adam_epsilon)

    def get_optimizer_state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def load_optimizer_state_dict(self, state_dict: dict):
        self._optimizer.load_state_dict(state_dict=state_dict)

    def _train_on_batch(self, batch):
        self._optimizer.zero_grad()
        loss, log_dict = self._objective.compute_losses(*batch)
        log_dict = OrderedDict({"model/{}".format(k): v for k, v in log_dict.items()})
        loss.backward()
        if self._clip_grad_by_norm:
            _ = nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_norm)
        self._optimizer.step()
        log_dict = OrderedDict({"model/neg_elbo": loss.detach_().cpu().numpy(), **log_dict})
        return log_dict


