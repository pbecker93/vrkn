import time
import torch
from collections import OrderedDict
from ssm_mbrl.ssm_interface.abstract_objective import AbstractObjective
from ssm_mbrl.ssm_interface.abstract_ssm import AbstractSSM


data = torch.utils.data


class AbstractTrainer:

    def __init__(self,
                 objective: AbstractObjective,
                 model: AbstractSSM,
                 eval_interval: int):
        self._objective = objective
        self._model = model
        self._eval_interval = eval_interval

    def _train_on_batch(self, batch) -> OrderedDict:
        raise NotImplementedError()

    def train_epoch(self, data_loader: data.DataLoader) -> tuple[OrderedDict, float]:
        batches_per_epoch = len(data_loader)
        avg_log_dict = None
        t0 = time.time()
        for i, batch in enumerate(data_loader):
            log_dict = self._train_on_batch(batch)
            if i == 0:
                avg_log_dict = OrderedDict({k: v / batches_per_epoch for k, v in log_dict.items()})
            else:
                for k, v in log_dict.items():
                    avg_log_dict[k] += v / batches_per_epoch

        return avg_log_dict, time.time() - t0

    def will_evaluate(self, iteration: int) -> bool:
        return self._eval_interval > 0 and iteration % self._eval_interval == 0

    def evaluate(self,
                 iteration: int,
                 data_loader: data.DataLoader) -> OrderedDict:
        # We still set to train here as this is to measure the train loss (on a validation/test set) for evaluations
        # on different metrics use another class.
        if self.will_evaluate(iteration):
            batches_per_epoch = len(data_loader)
            avg_log_dict = None
            for i, batch in enumerate(data_loader):
                loss, log_dict = self._objective.compute_losses(*batch)
                log_dict = OrderedDict({"model/{}".format(k): v for k, v in log_dict.items()})
                log_dict = OrderedDict({"model/neg_elbo": loss.detach_().cpu().numpy(), **log_dict})
                if i == 0:
                    avg_log_dict = OrderedDict({k: v / batches_per_epoch for k, v in log_dict.items()})
                else:
                    for k, v in log_dict.items():
                        avg_log_dict[k] += v / batches_per_epoch
            return avg_log_dict
        else:
            # TODO check returns
            return OrderedDict()

    def get_optimizer_state_dict(self) -> dict:
        raise NotImplementedError

    def load_optimizer_state_dict(self, state_dict: dict):
        raise NotImplementedError
