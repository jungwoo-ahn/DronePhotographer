"""VLADroneDataset — v7 windows for the VLA baseline.

Thin wrapper over `common.dataset_base.BasePolicyDataset` (same windows, same HER
goal sampling, same clamped-goal filter as the Cosmos dataset — so the two train
on the identical sample distribution). Emits only what the VLA needs:

  - `state_image`: (3, H, W) in [-1, 1] — the window's start frame.
  - `goal_vec`:    (D_goal,) normalized to [-1, 1].
  - `action_chunk`: (chunk_size, 5) normalized.
  - `meta`: dict for debugging.

Dropped vs CosmosDroneDataset: `next_state_image` (no world model → no
future-frame target) and `value_target` (no value head). The VLM's own image
processor turns `state_image` into pixel tensors at train time (in the policy's
`prepare_inputs`), so we keep the image as a plain [-1, 1] CHW tensor here.
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
    """Load JPEG/PNG, resize to (H, W), return (3, H, W) in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (target_resolution[1], target_resolution[0]):
        img = img.resize((target_resolution[1], target_resolution[0]), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class VLADroneDataset(Dataset):
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
        if len(self.base):
            p = Path(self.base[0].start.image)
            if not p.exists():
                raise FileNotFoundError(f"missing rendered image: {p}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        s = self.base[idx]
        img = _load_image_as_tensor(Path(s.start.image), self.target_resolution)
        goal = s.goal_vec
        if self.normalize:
            goal = normalize_goal(goal, self.base.goal_keys)
        action_chunk = s.action_chunk
        if self.normalize_actions:
            action_chunk = normalize_action_5d(action_chunk, self.action_scale)
        return {
            "state_image": img,
            "goal_vec": torch.from_numpy(goal),
            "action_chunk": torch.from_numpy(action_chunk),
            "meta": {
                "annotation_path": str(s.start.annotation_path),
                "pair_idx": s.start.pair_idx,
                "start_frame_idx": s.start.frame_idx,
                "goal_frame_idx": s.goal.frame_idx,
                "scene": s.start.scene,
                "object": s.start.object,
            },
        }


__all__ = ["VLADroneDataset"]
