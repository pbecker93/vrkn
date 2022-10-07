import os
from typing import Optional
import torch
import random
import numpy as np
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor
data = torch.utils.data


class ReplayBuffer:

    def __init__(self,
                 add_reward_to_obs: bool,
                 obs_are_images: list[bool],
                 img_preprocessor: Optional[ImgPreprocessor],
                 dataloader_num_workers: int):

        self._add_reward_to_obs = add_reward_to_obs
        self._all_data = {}
        self._obs_means = None
        self._obs_stds = None

        self._obs_are_images = obs_are_images

        self._has_loss_masks = False
        self._has_obs_valid = False

        self._img_preprocessor = img_preprocessor
        self._frozen = False

        self._dataloader_num_workers = dataloader_num_workers

    def save_normalization_parameters(self,
                                      obs_means: list[torch.Tensor],
                                      obs_stds: list[torch.Tensor]):
        self._obs_means = obs_means
        self._obs_stds = obs_stds

    @property
    def normalization_parameters(self):
        return self._obs_means, self._obs_stds

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def add_data(self,
                 key: str,
                 observations: list[list[torch.Tensor]],
                 actions: list[torch.Tensor],
                 rewards: list[torch.Tensor],
                 infos: list[dict]):
        assert not self.is_frozen
        self._all_data[key] = {"obs": observations,
                               "actions": actions,
                               "rewards": rewards,
                               "infos": infos}

        self._has_loss_masks = self._has_loss_masks or "loss_mask" in infos[0].keys()
        self._has_obs_valid = self._has_obs_valid or "obs_valid" in infos[0].keys()

    @property
    def all_keys(self):
        return self._all_data.keys()

    @property
    def has_loss_masks(self) -> bool:
        return self._has_loss_masks

    @property
    def has_obs_valid(self) -> bool:
        return self._has_obs_valid

    def _collect_values(self,
                        use_keys: Optional[list[str]] = None) \
            -> tuple[list, list, list, Optional[list], Optional[list]]:
        all_obs, all_actions, all_rewards = [], [], []
        all_loss_masks = [] if self._has_loss_masks else None
        all_obs_valid = [] if self._has_obs_valid else None
        for k, v in self._all_data.items():
            if use_keys is None or k in use_keys:
                all_obs += v["obs"]
                all_actions += v["actions"]
                all_rewards += v["rewards"]
                if self._has_loss_masks:
                    all_loss_masks += [d["loss_mask"] for d in v["infos"]]
                if self._has_obs_valid:
                    all_obs_valid += [d["obs_valid"] for d in v["infos"]]
        return all_obs, all_actions, all_rewards, all_loss_masks, all_obs_valid

    def _collect_infos(self, use_keys: Optional[list[str]] = None) -> list:
        all_infos = []
        for k, v in self._all_data.items():
            if use_keys is None or k in use_keys:
                all_infos += v["infos"]
        return all_infos

    def get_data_loader(self,
                        device: torch.device,
                        batch_size: int,
                        num_batches: int,
                        seq_length: int,
                        use_keys: Optional[list[str]] = None):
        obs, acts, rewards, loss_masks, obs_valid, shuffle_train_set =\
            self._sample_seqs(batch_size=batch_size,
                              num_batches=num_batches,
                              seq_length=seq_length,
                              use_keys=use_keys)

        obs = [torch.stack([o[i] for o in obs], dim=0) for i in range(len(obs[0]))]
        actions = torch.stack(acts, dim=0)
        rewards = torch.stack(rewards, dim=0)
        if loss_masks is None:
            loss_masks = [None] * len(obs)
        else:
            loss_masks = [None if loss_masks[0][i] is None else torch.stack([lm[i] for lm in loss_masks], dim=0)
                          for i in range(len(loss_masks[0]))]

        in_obs = obs
        if self._add_reward_to_obs:
            in_obs += [rewards]

        if obs_valid is None:
            obs_valid = [None] * (len(obs))
        else:
            obs_valid = [torch.stack([ov[i] for ov in obs_valid], dim=0) for i in range(len(obs_valid[0]))]
        return self._build_dataloader(batch_size=batch_size,
                                      device=device,
                                      obs=obs,
                                      rewards=rewards,
                                      prev_actions=actions,
                                      loss_masks=loss_masks + [None],
                                      obs_valid=obs_valid)

    def _sample_seqs(self,
                     batch_size: int,
                     num_batches: int,
                     seq_length: int,
                     use_keys: Optional[list[str]] = None):

        all_obs, all_actions, all_rewards, all_loss_masks, all_obs_valid = \
            self._collect_values(use_keys=use_keys)

        num_samples = batch_size * num_batches
        seq_idx = np.random.randint(low=0, high=len(all_actions), size=num_samples)
        obs, acts, rewards = [], [], []
        loss_masks = None if all_loss_masks is None else []
        obs_valid = None if all_obs_valid is None else []

        for si in seq_idx:
            start_idx = np.random.randint(low=0, high=all_actions[si].shape[0] - seq_length)
            time_slice = slice(start_idx, start_idx + seq_length)
            obs.append([o[time_slice] for o in all_obs[si]])
            acts.append(all_actions[si][time_slice])
            rewards.append(all_rewards[si][time_slice])
            if all_loss_masks is not None:
                loss_masks.append([None if lm is None else lm[time_slice] for lm in all_loss_masks[si]])
            if all_obs_valid is not None:
                obs_valid.append([None if ov is None else ov[time_slice] for ov in all_obs_valid[si]])
        return obs, acts, rewards, loss_masks, obs_valid, False

    def _build_dataloader(self,
                          batch_size: int,
                          device: torch.device,
                          obs: list[torch.Tensor],
                          rewards: torch.Tensor,
                          prev_actions: torch.Tensor,
                          loss_masks: Optional[list[Optional[torch.Tensor]]] = None,
                          obs_valid: Optional[list[torch.Tensor]] = None):

        idx_dict = {}
        data_tensors = []

        for i, o in enumerate(obs):
            idx_dict["obs_{}".format(i)] = len(data_tensors)
            data_tensors.append(o)

        idx_dict["rewards"] = len(data_tensors)
        data_tensors.append(rewards)

        idx_dict["actions"] = len(data_tensors)
        data_tensors.append(prev_actions)

        if loss_masks is not None:
            for i, lm in enumerate(loss_masks):
                if lm is not None:
                    idx_dict["loss_mask_{}".format(i)] = len(data_tensors)
                    data_tensors.append(lm)
        if obs_valid is not None:
            for i, ov in enumerate(obs_valid):
                if ov is not None:
                    idx_dict["obs_valid_{}".format(i)] = len(data_tensors)
                    data_tensors.append(ov)

        data_set = data.TensorDataset(*data_tensors)

        def collate_fn(data_batch):
            return self._collate_fn(data_batch=data_batch,
                                    num_obs=len(obs),
                                    idx_dict=idx_dict,
                                    device=device)

        def seed_worker(worker_id):
            worker_seed = (worker_id + torch.initial_seed()) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return data.DataLoader(data_set,
                               shuffle=False,
                               batch_size=batch_size,
                               num_workers=self._dataloader_num_workers,
                               worker_init_fn=seed_worker,
                               collate_fn=collate_fn)

    def _collate_fn(self,
                    data_batch: list[torch.Tensor],
                    num_obs: int,
                    idx_dict: dict,
                    device: torch.device):
        data_batch = data._utils.collate.default_collate(data_batch)

        if "cuda" in device.type:
            data_batch = data._utils.pin_memory.pin_memory(data_batch)

        obs_batch = []
        for i in range(num_obs):
            obs = data_batch[idx_dict["obs_{}".format(i)]].to(device, non_blocking=True)
            obs_batch.append(obs)

        prev_action_batch = data_batch[idx_dict["actions"]].to(device, non_blocking=True)
        rewards = data_batch[idx_dict["rewards"]].to(device, non_blocking=True)

        if self._has_loss_masks:
            loss_masks = []
            for i in range(num_obs + 1):
                k = "loss_mask_{}".format(i)
                if k in idx_dict.keys():
                    loss_masks.append(data_batch[idx_dict[k]].to(device, non_blocking=True))
                else:
                    loss_masks.append(None)
        else:
            loss_masks = None

        if self._has_obs_valid:
            obs_valid = []
            for i in range(num_obs):
                k = "obs_valid_{}".format(i)
                if k in idx_dict.keys():
                    obs_valid.append(data_batch[idx_dict[k]].to(device, non_blocking=True))
                else:
                    obs_valid.append(None)
        else:
            obs_valid = None

        for i, obs in enumerate(obs_batch):
            if self._obs_are_images[i]:
                obs_batch[i] = self._img_preprocessor(obs)

        return obs_batch, obs_batch + [rewards], prev_action_batch, obs_valid, loss_masks
