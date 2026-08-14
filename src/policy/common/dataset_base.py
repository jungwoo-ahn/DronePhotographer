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
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass

from src.policy.common.action_repr import ACTION_DIM, POSE_DIM, encode_action_5d
from src.policy.common.annotations import (
    TrajectoryWindow,
    ViewRecord,
    iter_goal_start_windows,
    iter_multiscale_windows,
    iter_windows,
    list_annotation_files,
    shoot_column,
)
from src.policy.common.goal_space import goal_keys, goal_vector
from src.policy.common.reward import pose_distance_value

GOAL_SAMPLING_MODES = ("uniform_future", "end")
SAMPLING_SCHEMES = ("sliding_window", "multiscale_bidir", "goal_start")

# Bump when the on-disk index layout changes (e.g. _Entry / ViewRecord fields,
# window enumeration, or action encoding) so stale caches are ignored, not
# silently reused. The key already folds in every config knob + the annotation
# files' size/mtime; this covers code changes those can't see.
_INDEX_CACHE_VERSION = "2"  # v2: 6D action (pose+shoot) + goal_start sampler


def _index_cache_key(
    files: Sequence[Path],
    chunk_size: int,
    stride: int,
    sampling_scheme: str,
    offsets: Sequence[int],
    goal_keys_: Sequence[str],
    filter_clamped_goals: bool,
    goal_sampling: str,
    val_pair_stride: int,
    val_split_level: str,
    val_names: Sequence[str] | None,
    split: str,
    max_samples: int | None,
    goal_start_max_per_pair: int = 0,
    goal_start_seed: int = 0,
) -> str:
    """Stable digest of everything that shapes the built window index.

    Includes each annotation file's size + mtime, so regenerating the data (new
    renders, re-scored profiles) busts the cache without a manual purge.
    """
    h = hashlib.sha256()
    h.update(_INDEX_CACHE_VERSION.encode())
    params = [
        int(chunk_size), int(stride), str(sampling_scheme), tuple(int(o) for o in offsets),
        tuple(goal_keys_), bool(filter_clamped_goals), str(goal_sampling),
        int(val_pair_stride), str(val_split_level),
        tuple(sorted(val_names)) if val_names is not None else None,
        str(split), None if max_samples is None else int(max_samples),
        int(goal_start_max_per_pair), int(goal_start_seed),
    ]
    h.update(repr(params).encode())
    for f in sorted(str(x) for x in files):
        try:
            st = os.stat(f)
            h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(f"{f}:missing".encode())
    return h.hexdigest()[:16]


def _entry_views(entry: "_Entry"):
    """Every ViewRecord an entry references (with duplicates — stripping is idempotent)."""
    w = entry.window
    yield w.start
    yield w.end
    yield from (w.intermediate or [])
    yield from (getattr(w, "future", None) or [])
    yield from (getattr(w, "keyframes", None) or [])
    for view, _ in entry.candidates:
        yield view


def _load_or_build_index(
    cache_dir: str | Path | None,
    key: str,
    build_fn: Callable[[], list["_Entry"]],
) -> list["_Entry"]:
    """Return the window index from disk cache if present, else build + cache it.

    On a miss the built index has its per-ViewRecord `raw` score dicts dropped
    (consumed at build time only — goal_vector / clamp filter — never at
    __getitem__), which both shrinks the pickle and frees the RAM. The write is
    atomic (temp + os.replace) so concurrent DDP ranks never read a partial file.
    """
    if cache_dir is None:
        return build_fn()
    cache_dir = Path(cache_dir)
    path = cache_dir / f"win_index_{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # corrupt / unreadable -> rebuild and overwrite
    entries = build_fn()
    for e in entries:
        for vr in _entry_views(e):
            vr.raw = {}
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(entries, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except (OSError, NameError):
            pass
    return entries


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
        out[i, :POSE_DIM] = encode_action_5d(
            np.asarray(prev.camera_position, dtype=np.float32),
            np.asarray(prev.camera_forward, dtype=np.float32),
            np.asarray(prev.camera_up, dtype=np.float32),
            np.asarray(nxt.camera_position, dtype=np.float32),
            np.asarray(nxt.camera_forward, dtype=np.float32),
            np.asarray(nxt.camera_up, dtype=np.float32),
        )
    # Dim 5: the latched shoot channel (0 before goal arrival, 1 from it on). For
    # goal_start windows it fires when the clamped walk reaches goal_frame; for
    # sliding/multiscale (no goal_frame, goal at boundary) it is all-zero by design.
    out[:, POSE_DIM] = shoot_column(window)
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
        goal_start_max_per_pair: int = 24,
        goal_start_seed: int = 0,
        val_pair_stride: int = 0,
        val_split_level: str = "pair",
        val_names: Sequence[str] | None = None,
        split: str = "train",
        cache_dir: str | Path | None = None,
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
        self.goal_start_max_per_pair = int(goal_start_max_per_pair)
        self.goal_start_seed = int(goal_start_seed)
        self._files = list_annotation_files(annotation_roots)

        def _windows(f):
            if sampling_scheme == "multiscale_bidir":
                return iter_multiscale_windows(f, chunk_size=chunk_size, offsets=self.offsets)
            if sampling_scheme == "goal_start":
                return iter_goal_start_windows(
                    f, chunk_size=chunk_size,
                    max_per_pair=self.goal_start_max_per_pair, seed=self.goal_start_seed,
                )
            return iter_windows(f, chunk_size=chunk_size, stride=stride)

        def _build() -> list[_Entry]:
            entries: list[_Entry] = []
            for f in self._files:
                for window in _windows(f):
                    if val_pair_stride > 0 or self._val_names is not None:
                        is_val = _is_val_pair(
                            window.annotation_path, window.pair_idx, val_pair_stride,
                            val_split_level, self._val_names,
                        )
                        if is_val != (split == "val"):
                            continue
                    # goal_start carries an explicit well-framed goal frame; multiscale_bidir
                    # pins the goal to the endpoint (empty future); sliding_window uses the
                    # HER pool per goal_sampling.
                    if sampling_scheme == "goal_start":
                        pool = [window.goal_frame]
                    elif goal_sampling == "end" or sampling_scheme == "multiscale_bidir":
                        pool = [window.end]
                    else:
                        pool = [window.end, *window.future]
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
                    entries.append(_Entry(window, _compute_action_chunk(window), candidates))
                    if max_samples and len(entries) >= max_samples:
                        return entries
                if max_samples and len(entries) >= max_samples:
                    return entries
            return entries

        # Building the index is the slow part: multiscale_bidir enumerates ~5M
        # strided windows single-threaded (~1h). Cache it keyed on every
        # index-shaping param + each annotation file's size/mtime, so reruns and
        # extra DDP ranks load it in seconds. cache_dir=None disables caching.
        self._entries: list[_Entry] = _load_or_build_index(
            cache_dir,
            _index_cache_key(
                self._files, chunk_size, stride, sampling_scheme, self.offsets,
                self.goal_keys, filter_clamped_goals, goal_sampling,
                val_pair_stride, val_split_level, self._val_names, split, max_samples,
                self.goal_start_max_per_pair, self.goal_start_seed,
            ),
            _build,
        )

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
