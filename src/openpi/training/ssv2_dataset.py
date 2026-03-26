"""SS-v2 dataset for physical understanding pre-training (Stage 1-A: pure A2L).

Training signal (Action-to-Language):
  Input:  3 video frames (start, middle, end) — no label provided to the model
  Output: correct action description (label)

This forces the model to generate a description from visual content alone.
There is no trivial shortcut: the correct answer is never in the input.

LACY correspondence:
  This trains the A2L (Action-to-Language) capability used by the L2C verifier in Stage 2.

Previous designs (deprecated):
  v1 - instruction reproduction: model copied input label (trivial solution)
  v2 - Yes/No + description: "always yes" collapse (pos=1 token easy, neg=many tokens hard)

Input modes (ablation via `input_mode`):
  "3frame" : start -> base_0_rgb, middle -> base_1_rgb, end -> base_2_rgb  [default]
  "2frame" : start -> base_0_rgb, end   -> base_1_rgb                      [baseline]
  "start"  : start -> base_0_rgb only
  "end"    : end   -> base_0_rgb only
"""

import json
from pathlib import Path
from typing import Literal, SupportsIndex

import cv2
import numpy as np

_IMAGE_H = 224
_IMAGE_W = 224
_ACTION_DIM = 32
_ACTION_HORIZON = 32

InputMode = Literal["3frame", "2frame", "start", "end"]


class SSv2Dataset:
    """Map-style dataset for SS-v2 A2L pre-training.

    Returns a dict compatible with openpi's transform pipeline:
        image:          dict of {camera_key: (H, W, 3) uint8}
        image_mask:     dict of {camera_key: True/False}
        state:          (action_dim,) float32 zeros
        target_prompt:  str  -- correct action description (the generation target)
        actions:        (action_horizon, action_dim) float32 zeros (dummy)
    """

    def __init__(
        self,
        labels_path: str | Path,
        frames_dir: str | Path,
        input_mode: InputMode = "3frame",
        action_dim: int = _ACTION_DIM,
        action_horizon: int = _ACTION_HORIZON,
    ):
        self._frames_dir = Path(frames_dir)
        self._input_mode = input_mode
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        with open(labels_path) as f:
            self._entries = json.load(f)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: SupportsIndex) -> dict:
        entry = self._entries[index.__index__()]
        video_id = entry["id"]
        images, image_masks = self._load_images(video_id)

        return {
            "image": images,
            "image_mask": image_masks,
            "state": np.zeros(self._action_dim, dtype=np.float32),
            "target_prompt": entry["label"],
            "actions": np.zeros((self._action_horizon, self._action_dim), dtype=np.float32),
        }

    def _load_images(self, video_id: str) -> tuple[dict, dict]:
        if self._input_mode == "3frame":
            images = {
                "base_0_rgb": self._load_frame(video_id, "start"),
                "base_1_rgb": self._load_frame(video_id, "middle"),
                "base_2_rgb": self._load_frame(video_id, "end"),  # not "wrist_*" to get full augmentation
            }
            image_masks = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "base_2_rgb": np.True_,
            }
        elif self._input_mode == "2frame":
            images = {
                "base_0_rgb": self._load_frame(video_id, "start"),
                "base_1_rgb": self._load_frame(video_id, "end"),
            }
            image_masks = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "wrist_0_rgb": np.False_,
            }
        elif self._input_mode == "start":
            images = {"base_0_rgb": self._load_frame(video_id, "start")}
            image_masks = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.False_,
                "wrist_0_rgb": np.False_,
            }
        elif self._input_mode == "end":
            images = {"base_0_rgb": self._load_frame(video_id, "end")}
            image_masks = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.False_,
                "wrist_0_rgb": np.False_,
            }
        else:
            raise ValueError(f"Unknown input_mode: {self._input_mode!r}")
        return images, image_masks

    def _load_frame(self, video_id: str, which: str) -> np.ndarray:
        path = self._frames_dir / f"{video_id}_{which}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (_IMAGE_H, _IMAGE_W):
            img = cv2.resize(img, (_IMAGE_W, _IMAGE_H), interpolation=cv2.INTER_LINEAR)
        return img
