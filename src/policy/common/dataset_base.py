"""Base Dataset over v7 trajectory windows.

Each sample is a K-step window from one of a placement's `accepted_pairs[i].trajectory_32f`:
the (start_frame, end_frame, chunk_size · ACTION_DIM action chunk, hindsight-relabeled
goal). Subclasses (`src/policy/cosmos/dataset.py`) shape the per-sample dict
into model-specific tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass

from src.policy.common.action_repr import ACTION_DIM, encode_action_5d
from src.policy.common.annotations import (
    TrajectoryWindow,
    ViewRecord,
    iter_windows,
    list_annotation_files,
)
from src.policy.common.goal_space import goal_keys, goal_vector
from src.policy.common.reward import pose_distance_value


def _is_clamped(scores: dict) -> bool:
    """True if the frame hit the scorer's off-screen sentinel (bbox keys zeroed).

    `compute_v5_scores` zeroes the bbox-derived keys when the full projection
    blows past its 4x sanity clamp — a VLM-era sentinel meaning "no meaningful
    framing", not a measurement. A goal profile in that state is garbage to
    condition on, so windows ending on such frames are filtered out.
    """
    return scores.get("occupancy") == 0 and scores.get("bbox_y_offset") == 0


@dataclass
class Sample:
    """One window with a K-step action chunk and a hindsight-relabeled goal."""

    start: ViewRecord
    end: ViewRecord
    intermediate: list[ViewRecord]
    action_chunk: np.ndarray              # (chunk_size, ACTION_DIM)
    goal_vec: np.ndarray                  # (D_goal,)
    value: float                          # -geometric distance from start profile to goal
    chunk_size: int


def _compute_action_chunk(window: TrajectoryWindow) -> np.ndarray:
    frames = [window.start, *window.intermediate, window.end]
    out = np.zeros((window.chunk_size, ACTION_DIM), dtype=np.float32)
    for i in range(window.chunk_size):
        prev = frames[i]
        nxt = frames[i + 1]
        out[i] = encode_action_5d(
            np.asarray(prev.camera_position, dtype=np.float32),
            np.asarray(prev.camera_forward, dtype=np.float32),
            np.asarray(prev.camera_up, dtype=np.float32),
            np.asarray(nxt.camera_position, dtype=np.float32),
            np.asarray(nxt.camera_forward, dtype=np.float32),
            np.asarray(nxt.camera_up, dtype=np.float32),
        )
    return out


class BasePolicyDataset(Dataset):
    """Indexable list of v7 trajectory-window samples.

    Args:
      annotation_roots: list of files (`data.json`) or directories
        (recursively globs `**/data.json`).
      goal_score_keys: subset of V5 keys to use as the goal vector. Default = all 8.
      chunk_size: number of actions per sample (= temporal extent of the window).
      stride: window stride along each 32-frame trajectory.
      max_samples: optional cap (smoke tests).
      filter_clamped_goals: drop windows whose goal (end) frame hit the scorer's
        off-screen sentinel — its profile is a fabricated "zero-size subject at
        (0,0)", useless as a conditioning goal. (~31% of windows on real v7 data.)
    """

    def __init__(
        self,
        annotation_roots: Sequence[str | Path],
        *,
        goal_score_keys: Sequence[str] | None = None,
        chunk_size: int = 8,
        stride: int = 1,
        max_samples: int | None = None,
        filter_clamped_goals: bool = True,
    ) -> None:
        self.goal_keys = goal_keys(goal_score_keys)
        self.chunk_size = chunk_size
        self.stride = stride
        self.filter_clamped_goals = filter_clamped_goals
        self._files = list_annotation_files(annotation_roots)
        self._samples: list[Sample] = []
        for f in self._files:
            for window in iter_windows(f, chunk_size=chunk_size, stride=stride):
                g = goal_vector(window.end.raw, self.goal_keys)
                if not np.isfinite(g).all():
                    continue
                if self.filter_clamped_goals and _is_clamped(window.end.raw):
                    continue
                # Value = -(geometric distance from the START pose to the GOAL = END
                # pose). Computed from camera poses + subject geometry, NOT from the
                # bbox-derived score pixels — poses are exact for every frame, with
                # no off-screen sentinel to corrupt size/aim.
                value = pose_distance_value(
                    window.start.camera_position, window.start.camera_forward, window.start.camera_up,
                    window.end.camera_position, window.end.camera_forward, window.end.camera_up,
                    subject_center=window.start.subject_center,
                    subject_height=window.start.subject_height,
                )
                self._samples.append(Sample(
                    start=window.start,
                    end=window.end,
                    intermediate=window.intermediate,
                    action_chunk=_compute_action_chunk(window),
                    goal_vec=g,
                    value=value,
                    chunk_size=window.chunk_size,
                ))
                if max_samples and len(self._samples) >= max_samples:
                    return
            if max_samples and len(self._samples) >= max_samples:
                return

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]
