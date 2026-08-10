"""Base Dataset over v7 trajectory windows.

Each sample is a K-step window from one of a placement's `accepted_pairs[i].trajectory_32f`:
the (start_frame, end_frame, chunk_size · ACTION_DIM action chunk, hindsight-relabeled
goal). Subclasses (`src/policy/cosmos/dataset.py`) shape the per-sample dict
into model-specific tensors.

Goal relabeling (HER-"future"): the goal profile is NOT pinned to the window's
end frame. With `goal_sampling="uniform_future"` (default), each `__getitem__`
draws the goal frame uniformly from [end_frame, last_frame] of the same
trajectory — the action chunk and next-frame target stay anchored to the window
(they are the *consequence* of the actions), while the goal vector and value
target follow the drawn frame. This decouples the goal horizon from the action
horizon: the conditioner and value head see goals at every distance 8..31
steps out, matching inference-time goals that are not 8 steps away.
`goal_sampling="end"` restores the legacy fixed-offset behavior.

Randomness uses the global numpy RNG, which torch's DataLoader re-seeds per
worker per epoch — repeated `__getitem__(i)` calls may return different goals
by design.
"""

from __future__ import annotations

import hashlib
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
    iter_multiscale_windows,
    iter_windows,
    list_annotation_files,
)
from src.policy.common.goal_space import goal_keys, goal_vector
from src.policy.common.reward import pose_distance_value

GOAL_SAMPLING_MODES = ("uniform_future", "end")
SAMPLING_SCHEMES = ("sliding_window", "multiscale_bidir")


VAL_SPLIT_LEVELS = ("pair", "placement", "scene", "object")


def _split_key(annotation_path: Path, pair_idx: int, level: str) -> str:
    """The name that decides a window's split side, at the given unit level.

    The unit (`level`) sets what the val metric measures generalization to:
      pair       — new camera trajectories in a seen scene+object (dev default)
      placement  — new scene__object combinations
      scene      — unseen environments (placement dirs are "<scene>__<object>")
      object     — unseen subjects

    Never split below "pair": overlapping windows share frames and would leak.
    """
    placement = annotation_path.parent.name
    if level == "pair":
        return f"{placement}:{pair_idx}"
    if level == "placement":
        return placement
    if level == "scene":
        return placement.split("__")[0]
    if level == "object":
        return placement.split("__")[-1]
    raise ValueError(f"val_split_level must be one of {VAL_SPLIT_LEVELS}, got {level!r}")


def _is_val_pair(
    annotation_path: Path,
    pair_idx: int,
    val_pair_stride: int,
    level: str = "pair",
    val_names: frozenset[str] | None = None,
) -> bool:
    """Split-side assignment for one trajectory pair.

    With `val_names` (a frozen manifest): val iff the unit's name is listed —
    a pinned val set that never changes; all future arrivals are train.

    Otherwise, deterministic hash: ~1/val_pair_stride of <level> units go to
    val. Assignment depends only on the unit's own name — adding, removing, or
    reordering other data never flips an existing item's side. (Renaming an
    item, or changing stride/level, redefines the split.)
    """
    key = _split_key(annotation_path, pair_idx, level)
    if val_names is not None:
        return key in val_names
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return h % val_pair_stride == 0


def _is_clamped(scores: dict) -> bool:
    """True if the frame hit the scorer's off-screen sentinel (bbox keys zeroed).

    `compute_v5_scores` zeroes the bbox-derived keys when the full projection
    blows past its 4x sanity clamp — a VLM-era sentinel meaning "no meaningful
    framing", not a measurement. A goal profile in that state is garbage to
    condition on, so such frames are excluded from the goal candidate pool.
    """
    return scores.get("occupancy") == 0 and scores.get("bbox_y_offset") == 0


@dataclass
class Sample:
    """One window with a K-step action chunk and a hindsight-relabeled goal."""

    start: ViewRecord
    end: ViewRecord
    intermediate: list[ViewRecord]
    action_chunk: np.ndarray              # (chunk_size, ACTION_DIM)
    goal_vec: np.ndarray                  # (D_goal,) — profile of `goal`, not necessarily `end`
    value: float                          # -geometric distance from start pose to goal pose
    chunk_size: int
    goal: ViewRecord                      # the frame whose profile is the goal (== end in "end" mode)


@dataclass
class _Entry:
    """One window with its precomputed action chunk and valid goal candidates."""

    window: TrajectoryWindow
    action_chunk: np.ndarray
    candidates: list[tuple[ViewRecord, np.ndarray]]   # (goal frame, goal_vec)


def _compute_action_chunk(window: TrajectoryWindow) -> np.ndarray:
    # keyframes are the chunk_size+1 frames to encode between: consecutive for
    # sliding_window, strided for multiscale_bidir. Re-encoding between the strided
    # keyframes (not summing single-step deltas) is required — camera-local deltas
    # do not compose additively.
    frames = window.keyframes or [window.start, *window.intermediate, window.end]
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
      max_samples: optional cap on windows (smoke tests).
      filter_clamped_goals: exclude goal candidates that hit the scorer's
        off-screen sentinel — such a profile is a fabricated "zero-size subject
        at (0,0)", useless as a conditioning goal. Windows with no valid
        candidate at all are dropped.
      goal_sampling: "uniform_future" (default) draws the goal frame uniformly
        from [end_frame, last_frame] of the trajectory on every __getitem__;
        "end" pins it to the window's end frame (legacy fixed-offset behavior).
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
        goal_sampling: str = "uniform_future",
        sampling_scheme: str = "sliding_window",
        offsets: Sequence[int] = (8, 16, 24),
        val_pair_stride: int = 0,
        val_split_level: str = "pair",
        val_names: Sequence[str] | None = None,
        split: str = "train",
    ) -> None:
        if goal_sampling not in GOAL_SAMPLING_MODES:
            raise ValueError(f"goal_sampling must be one of {GOAL_SAMPLING_MODES}, got {goal_sampling!r}")
        if sampling_scheme not in SAMPLING_SCHEMES:
            raise ValueError(f"sampling_scheme must be one of {SAMPLING_SCHEMES}, got {sampling_scheme!r}")
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if split == "val" and val_pair_stride <= 0 and not val_names:
            raise ValueError("split='val' requires val_pair_stride > 0 or val_names")
        self._val_names = frozenset(val_names) if val_names else None
        self.goal_keys = goal_keys(goal_score_keys)
        self.chunk_size = chunk_size
        self.stride = stride
        self.filter_clamped_goals = filter_clamped_goals
        self.goal_sampling = goal_sampling
        self.sampling_scheme = sampling_scheme
        self.offsets = tuple(offsets)
        self._files = list_annotation_files(annotation_roots)
        self._entries: list[_Entry] = []

        def _windows(f):
            if sampling_scheme == "multiscale_bidir":
                return iter_multiscale_windows(f, chunk_size=chunk_size, offsets=self.offsets)
            return iter_windows(f, chunk_size=chunk_size, stride=stride)

        for f in self._files:
            for window in _windows(f):
                if val_pair_stride > 0 or self._val_names is not None:
                    is_val = _is_val_pair(
                        window.annotation_path, window.pair_idx, val_pair_stride,
                        val_split_level, self._val_names,
                    )
                    if is_val != (split == "val"):
                        continue
                # multiscale_bidir pins the goal to the endpoint (window.future is
                # empty); sliding_window uses the HER pool per goal_sampling.
                pool = [window.end] if (goal_sampling == "end" or sampling_scheme == "multiscale_bidir") \
                    else [window.end, *window.future]
                candidates: list[tuple[ViewRecord, np.ndarray]] = []
                for view in pool:
                    g = goal_vector(view.raw, self.goal_keys)
                    if not np.isfinite(g).all():
                        continue
                    if self.filter_clamped_goals and _is_clamped(view.raw):
                        continue
                    candidates.append((view, g))
                if not candidates:
                    continue
                self._entries.append(_Entry(window, _compute_action_chunk(window), candidates))
                if max_samples and len(self._entries) >= max_samples:
                    return
            if max_samples and len(self._entries) >= max_samples:
                return

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Sample:
        entry = self._entries[idx]
        window = entry.window
        j = int(np.random.randint(len(entry.candidates))) if len(entry.candidates) > 1 else 0
        goal_view, g = entry.candidates[j]
        # Value = -(geometric distance from the START pose to the GOAL pose).
        # Computed from camera poses + subject geometry, NOT from the
        # bbox-derived score pixels — poses are exact for every frame, with
        # no off-screen sentinel to corrupt size/aim.
        value = pose_distance_value(
            window.start.camera_position, window.start.camera_forward, window.start.camera_up,
            goal_view.camera_position, goal_view.camera_forward, goal_view.camera_up,
            subject_center=window.start.subject_center,
            subject_height=window.start.subject_height,
        )
        return Sample(
            start=window.start,
            end=window.end,
            intermediate=window.intermediate,
            action_chunk=entry.action_chunk,
            goal_vec=g,
            value=value,
            chunk_size=window.chunk_size,
            goal=goal_view,
        )
