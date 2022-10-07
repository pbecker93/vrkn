import os
import numpy as np
import gym
from typing import Optional


class OcclusionWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 vid_folder: Optional[str] = None,
                 temp_step: int = 1,
                 occluded_img_idx: int = 0):
        super(OcclusionWrapper, self).__init__(env=env)
        self._all_videos = [os.path.join(vid_folder, vid) for vid in os.listdir(vid_folder)]
        self._current_frames = None
        self._frame_idx = None
        self._temp_step = temp_step
        self._occluded_img_idx = occluded_img_idx

    def _add_occlusions(self, image):

        occ = self._current_frames[self._frame_idx]
        mask = np.logical_not(np.any(occ > 14, axis=2))
        uint_mask = mask.astype(np.uint8)[..., None]
        return uint_mask * image + (1 - uint_mask) * occ, mask

    def _reset_mask_generation(self):
        cur_vid_idx = np.random.randint(low=0, high=len(self._all_videos))
        self._current_frames = dict(np.load(self._all_videos[cur_vid_idx]))["frames"]

        #self._current_frames = raw_current_frames, (0, 3, 1, 2)))

        high = self._current_frames.shape[0] - self.env.max_seq_length * self._temp_step
        self._frame_idx = np.random.randint(low=0, high=high)

    def reset(self):
        self._reset_mask_generation()
        obs, info = self.env.reset()
        obs[self._occluded_img_idx], mask = self._add_occlusions(obs[self._occluded_img_idx])
        loss_masks = [None] * len(obs)
        loss_masks[self._occluded_img_idx] = mask
        info["loss_mask"] = loss_masks
        return obs, info

    def step(self, action: np.ndarray) -> tuple[list[np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self._occluded_img_idx], mask = self._add_occlusions(obs[self._occluded_img_idx])
        loss_masks = [None] * len(obs)
        loss_masks[self._occluded_img_idx] = mask
        info["loss_mask"] = loss_masks
        self._frame_idx += self._temp_step

        return obs, reward, terminated, truncated, info

