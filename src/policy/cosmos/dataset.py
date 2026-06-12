"""CosmosDroneDataset over v7 trajectory windows.

Wraps `BasePolicyDataset` and converts each `Sample` into tensors:

  - `state_image`: (3, H, W) float in [-1, 1] — the trajectory window's start frame.
  - `next_state_image`: (3, H, W) — the end frame (ALOHA-style T_img=2 supervision).
  - `goal_vec`: (D_goal,) — V5 score vector of the goal frame, normalized to [-1, 1].
    With `goal_sampling="uniform_future"` (default) the goal frame is drawn
    uniformly from [end_frame, last_frame] of the trajectory per __getitem__
    (HER-"future" relabeling); `"end"` pins it to the window's end frame.
  - `action_chunk`: (chunk_size, ACTION_DIM=5) — the K consecutive actions in the
    window, normalized per-dim to ~[-1, 1] via `action_repr.normalize_action_5d`
    (unless `normalize_actions=False`). Denormalize predictions with
    `action_repr.denormalize_action_5d` before applying them to a camera pose.
  - `value_target`: scalar — `−geometric_pose_distance(start_pose, goal_pose)`,
    the camera-subject geometric distance from the start view to the goal frame
    (0 at the goal, more negative the further away the goal framing is).
  - `meta`: dict with annotation_path, pair_idx, frame indices, scene, object.

v7 image paths are resolved into absolute paths by `iter_windows`, so this
dataset does not need an `image_root` argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.policy.common.action_repr import normalize_action_5d
from src.policy.common.dataset_base import BasePolicyDataset
from src.policy.common.goal_space import normalize_goal


def _load_image_as_tensor(image_path: Path, target_resolution: tuple[int, int]) -> torch.Tensor:
    """Load a JPEG/PNG, resize to `target_resolution=(H, W)`, return (3, H, W) in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (target_resolution[1], target_resolution[0]):
        img = img.resize((target_resolution[1], target_resolution[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class CosmosDroneDataset(Dataset):
    def __init__(
        self,
        annotation_roots: Sequence[str | Path],
        *,
        goal_score_keys: Sequence[str] | None = None,
        chunk_size: int = 8,
        stride: int = 1,
        max_samples: int | None = None,
        target_resolution: tuple[int, int] = (480, 720),
        normalize_goal_to_unit_cube: bool = True,
        normalize_actions: bool = True,
        action_scale=None,
        filter_clamped_goals: bool = True,
        goal_sampling: str = "uniform_future",
        val_pair_stride: int = 0,
        val_split_level: str = "pair",
        val_names: Sequence[str] | None = None,
        split: str = "train",
    ) -> None:
        self.target_resolution = target_resolution
        self.normalize = normalize_goal_to_unit_cube
        self.normalize_actions = normalize_actions
        self.action_scale = action_scale
        self.chunk_size = chunk_size
        self.base = BasePolicyDataset(
            annotation_roots,
            goal_score_keys=goal_score_keys,
            chunk_size=chunk_size,
            stride=stride,
            max_samples=max_samples,
            filter_clamped_goals=filter_clamped_goals,
            goal_sampling=goal_sampling,
            val_pair_stride=val_pair_stride,
            val_split_level=val_split_level,
            val_names=val_names,
            split=split,
        )
        # Sanity-check that the first sample's images exist
        if len(self.base):
            s0 = self.base[0]
            for view in (s0.start, s0.end):
                if not Path(view.image).exists():
                    raise FileNotFoundError(f"missing rendered image: {view.image}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        s = self.base[idx]
        start_img = _load_image_as_tensor(Path(s.start.image), self.target_resolution)
        end_img = _load_image_as_tensor(Path(s.end.image), self.target_resolution)
        goal = s.goal_vec
        if self.normalize:
            goal = normalize_goal(goal, self.base.goal_keys)
        action_chunk = s.action_chunk
        if self.normalize_actions:
            action_chunk = normalize_action_5d(action_chunk, self.action_scale)
        return {
            "state_image": start_img,
            "next_state_image": end_img,
            "goal_vec": torch.from_numpy(goal),
            "action_chunk": torch.from_numpy(action_chunk),
            "value_target": torch.tensor(s.value, dtype=torch.float32),
            "meta": {
                "annotation_path": str(s.start.annotation_path),
                "pair_idx": s.start.pair_idx,
                "start_frame_idx": s.start.frame_idx,
                "end_frame_idx": s.end.frame_idx,
                "goal_frame_idx": s.goal.frame_idx,
                "chunk_size": s.chunk_size,
                "scene": s.start.scene,
                "object": s.start.object,
            },
        }
