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

Sampling scheme: `sampling_scheme="multiscale_bidir"` replaces the sliding window
with bidirectional multi-scale endpoints — per start frame, one window per signed
offset ±o (o in `offsets`, e.g. ±8/±16/±24) whose endpoint exists. The goal is the
endpoint the actions actually reach, so the SAME start with DIFFERENT endpoints
yields DIFFERENT action targets, forcing the policy to condition on the goal (the
fix for actions collapsing to `f(state)`). See `iter_multiscale_windows`.

Value: `Sample.value` is a per-step sequence (chunk_size,) — `value[k]` is the
negative geometric distance from the pre-action keyframe `k` to the goal (value[0]
is the legacy START→GOAL scalar).

Randomness uses the global numpy RNG, which torch's DataLoader re-seeds per
worker per epoch — repeated `__getitem__(i)` calls may return different goals
by design (sliding_window only; multiscale pins the goal to the endpoint).
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass

from src.common.action_repr import ACTION_DIM, encode_action_9d
from src.common.annotations import (
    DEFAULT_DELTA_RANGE,
    DEFAULT_GOAL_OCCUPANCY_RANGE,
    DEFAULT_MIN_GOAL_VISIBLE_FRAC,
    TrajectoryWindow,
    ViewRecord,
    iter_goal_start_windows,
    iter_multiscale_windows,
    iter_windows,
    list_annotation_files,
)
from src.common.facing import sector8
from src.common.goal_space import SUBJECT_BEARING_KEY, goal_keys, goal_vector, normalize_goal
from src.common.reward import VALUE_SCALE, pose_distance_value

GOAL_SAMPLING_MODES = ("uniform_future", "end")
SAMPLING_SCHEMES = ("sliding_window", "multiscale_bidir", "goal_start")
DEFAULT_MULTISCALE_OFFSETS = (8, 16, 24)

# The trajectory tree everything current reads. It sits INSIDE the older one:
#
#   data/trajectories/                              <- 3932 original placements
#   data/trajectories/v7_stage2_renders_lookat075/  <- 7885 re-renders (this constant)
#
# 3931 of the 7885 names appear in both, so a script pointed at the parent silently
# reads the superseded renders under the right-looking names — same placement, camera
# aimed at mid-torso (look-at 0.500) and no per-object yaw. That is the pool the
# visible_frac gate and the yaw re-render were meant to replace. Nesting the new tree
# under the old one made the mistake invisible; import this instead of hardcoding.
DEFAULT_TRAJ_ROOT = "data/trajectories/v7_stage2_renders_lookat075"
LEGACY_TRAJ_ROOT = "data/trajectories"

# Objects dropped from BOTH splits — defective data, not a train/val split concern:
#   rp_posedplus_00068_18_100k — ~100x scale bug (subject_height 169 m vs a ~1.7 m
#       median; turntable bounding radius 91 m), so its camera distances and action
#       magnitudes are off-scale versus every other asset.
#   Girls-Hugs_5d7050d8 — two people embracing, so "the subject" and its facing are
#       ambiguous; it was also the lone front_az=180 outlier in the facing map.
# Keyed by OBJECT (placement dir after "__"), unlike `train_exclude_names` which
# matches whole placement dir names.
DEFAULT_EXCLUDE_OBJECTS = frozenset({
    "rp_posedplus_00068_18_100k",
    "Girls-Hugs_5d7050d8",
})

# What the value latent predicts at each of the chunk_size steps:
#   cost_to_go       — scalar −pose_distance(keyframe_k, goal) (geometric, pose-based,
#                      immune to the off-screen clamp). Committed to the geometric metric.
#   achieved_profile — the goal_dim shot profile actually realized at keyframe_k
#                      (profile-space "world model"; metric-agnostic, same space as the goal).
#   profile_delta    — goal_profile − achieved_profile(keyframe_k), per-key cost-to-go in
#                      the goal space (→ 0 as the framing reaches the goal).
# The two profile modes are bbox/score-derived, so they inherit the scorer's off-screen
# clamp on degenerate intermediate frames (the pose-based cost_to_go deliberately avoids it).
VALUE_TARGET_MODES = ("cost_to_go", "achieved_profile", "profile_delta")


def resolve_value_spec(value_target_mode: str, goal_dim: int) -> tuple[int, float]:
    """(per-step value_dim, value_scale) for a value_target_mode.

    cost_to_go → 1 scalar/step, normalized by VALUE_SCALE (pose distance in radians).
    achieved_profile / profile_delta → goal_dim/step, already in normalize_goal space
      (value_scale = 1.0 — no extra scaling; the value IS in the [-1,1] goal space).
    """
    if value_target_mode == "cost_to_go":
        return 1, float(VALUE_SCALE)
    if value_target_mode in ("achieved_profile", "profile_delta"):
        return int(goal_dim), 1.0
    raise ValueError(f"value_target_mode must be one of {VALUE_TARGET_MODES}, got {value_target_mode!r}")


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


def _window_object(window: TrajectoryWindow) -> str:
    """The window's object key — same identity used by val_split_level='object'."""
    return window.object or window.annotation_path.parent.name.split("__")[-1]


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
    value: np.ndarray                     # per-step value target (value[k] = state BEFORE action k):
                                          # (chunk_size,) for cost_to_go, (chunk_size, goal_dim) for
                                          # achieved_profile / profile_delta. See VALUE_TARGET_MODES.
    chunk_size: int
    goal: ViewRecord                      # the frame whose profile is the goal (== end in "end"/multiscale modes)


@dataclass
class _Entry:
    """One window and its valid goal candidates.

    The action chunk and value sequence are derived from the window LAZILY in
    __getitem__, not stored here. Computing them for every entry at init is the
    dominant cost (`_compute_action_chunk` is ~97% of construction) and pointless:
    the multiscale scheme has ~3M entries but a run consumes ~400k, so eager
    precompute would waste ~hours of init on samples never seen.
    """

    window: TrajectoryWindow
    candidates: list[tuple[ViewRecord, np.ndarray]]   # (goal frame, goal_vec)


def arrival_step(shoot, threshold: float = 0.5):
    """First step in a predicted chunk where the shoot channel crosses `threshold`.

    Returns that index, or None if the chunk never crosses. It is also the number of pose
    steps to EXECUTE before halting, which is the whole point of returning an index rather
    than a bool.

    The channel is latched (`shoot_column`), so a chunk reads "travel, travel, ..., arrived,
    arrived". Only the steps at and after the crossing are the "you are there" steps whose
    pose action is zero by construction; everything before it is travel the policy planned.
    Halting on `max(shoot) > threshold` without consulting the index therefore throws that
    travel away -- and it is not a rare corner: over the 103 declarations in
    `runs/eval/shoot_probe_iter40000.json` the first crossing was at step 5, 6 or 7 and
    never at 0, so the discarded travel averaged 5.8 of the 8 steps.
    """
    import numpy as np

    crossed = np.asarray(shoot, dtype=np.float32) > float(threshold)
    return int(np.argmax(crossed)) if bool(crossed.any()) else None


def shoot_column(window: TrajectoryWindow) -> np.ndarray:
    """The `shoot` channel for this window: (chunk_size,) float32, latched at arrival.

    A LATCHED STATE, not an event spike: 0 before the goal frame, 1 from it onward. This is
    the `DimType.GRIPPER` reading of the channel — a gripper is open or closed, it does not
    "fire" — and it means an off-by-one step does not erase the signal the way a single-step
    spike would.

    It pairs with a property the data already has. `iter_goal_start_windows` clamps the walk
    at the goal, so every keyframe past arrival IS the goal frame and its pose action is
    exactly zero. So `shoot=1` and `action=0` turn on together: "you are there, stop moving,
    take the photo." That pairing is asserted in the tests.

    Why this exists at all: the policy does not recognise arrival. Measured in
    docs/v4_session_changes.md section 11, the final chunk moves as much as the previous one
    (0.31 vs 0.25), so no threshold on action magnitude can separate "arrived" from "still
    travelling". Termination has to be something the policy DECLARES.

    Labels are free by hindsight — the goal is the achieved profile of `goal_frame`, so
    arrival is wherever the walk first reaches that frame. Chunks whose goal lies beyond the
    chunk are all zeros ("not there yet"); measured on the export, ~30% contain the arrival.

    Arrival is found IN `keyframes` rather than computed from frame indices. The walk is
    `idx[k] = start + direction*k` clamped at the goal (`annotations.iter_goal_start_windows`)
    and `direction` can be NEGATIVE — the goal is drawn with `delta = abs(g - s)`, so it may
    sit before the start. A signed `goal_idx - start_idx` is then negative and latches the
    whole chunk to 1: measured, that put shoot=1 on 62% of chunks against a true arrival rate
    of 30%, every one of them with a non-zero pose action. Reading the walk cannot drift from
    how the walk is built.
    """
    frames = window.keyframes if window.keyframes else [
        window.start, *window.intermediate, window.end]
    goal_idx = window.goal_frame.frame_idx
    k = next((i for i, f in enumerate(frames) if f.frame_idx == goal_idx), len(frames))
    return (np.arange(window.chunk_size) >= k).astype(np.float32)


def _compute_action_chunk(window: TrajectoryWindow) -> np.ndarray:
    """The chunk_size actions between consecutive keyframes.

    `window.keyframes` is `[start, …, end]` — contiguous for sliding_window,
    strided by `frame_step` for multiscale_bidir. Re-encoding the action between
    strided keyframes IS the correct "merge" of `frame_step` single steps into one
    action (camera-local deltas do NOT compose additively).
    """
    frames = window.keyframes if window.keyframes else [window.start, *window.intermediate, window.end]
    out = np.zeros((window.chunk_size, ACTION_DIM), dtype=np.float32)
    for i in range(window.chunk_size):
        prev = frames[i]
        nxt = frames[i + 1]
        out[i] = encode_action_9d(
            np.asarray(prev.camera_position, dtype=np.float32),
            np.asarray(prev.camera_forward, dtype=np.float32),
            np.asarray(prev.camera_up, dtype=np.float32),
            np.asarray(nxt.camera_position, dtype=np.float32),
            np.asarray(nxt.camera_forward, dtype=np.float32),
            np.asarray(nxt.camera_up, dtype=np.float32),
        )
    return out


def _compute_value_sequence(
    window: TrajectoryWindow,
    goal_view: ViewRecord,
    mode: str = "cost_to_go",
    goal_key_list: Sequence[str] | None = None,
) -> np.ndarray:
    """The per-step value target for the chunk_size states BEFORE each action.

    Alignment is "before each action" in all modes: index k uses keyframe[k]
    (value[0] = start-state; the terminal keyframe == goal is NOT included, so no
    entry is trivially zero). Modes (see VALUE_TARGET_MODES):

      cost_to_go       → (chunk_size,)  value[k] = -pose_distance(keyframe[k], goal)
                         (geometric, pose-based, raw radians — normalized by VALUE_SCALE
                         downstream in the dataset).
      achieved_profile → (chunk_size, goal_dim)  normalize_goal(profile(keyframe[k]))
      profile_delta    → (chunk_size, goal_dim)  normalize_goal(goal) - achieved

    The profile modes return values already in the [-1,1] normalize_goal space (no
    VALUE_SCALE applied). They are score-derived, so off-screen/clamped intermediate
    frames yield degenerate profiles (nan→0); the pose-based cost_to_go avoids this.
    """
    frames = window.keyframes if window.keyframes else [window.start, *window.intermediate, window.end]
    ref = frames[0]
    n = window.chunk_size

    if mode == "cost_to_go":
        out = np.zeros(n, dtype=np.float32)
        for k in range(n):
            s = frames[k]
            out[k] = pose_distance_value(
                s.camera_position, s.camera_forward, s.camera_up,
                goal_view.camera_position, goal_view.camera_forward, goal_view.camera_up,
                subject_center=ref.subject_center,
                subject_height=ref.subject_height,
            )
        return out

    keys = goal_keys(goal_key_list)
    obj = _window_object(window)          # subject_bearing_deg needs the asset's facing
    achieved = np.zeros((n, len(keys)), dtype=np.float32)
    for k in range(n):
        p = np.nan_to_num(goal_vector(frames[k].raw, keys, object_key=obj), nan=0.0)
        achieved[k] = normalize_goal(p, keys)
    if mode == "achieved_profile":
        return achieved
    if mode == "profile_delta":
        g = normalize_goal(
            np.nan_to_num(goal_vector(goal_view.raw, keys, object_key=obj), nan=0.0), keys
        )
        return (g[None, :] - achieved).astype(np.float32)
    raise ValueError(f"value_target_mode must be one of {VALUE_TARGET_MODES}, got {mode!r}")


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
        Ignored when sampling_scheme="multiscale_bidir".
      sampling_scheme: "sliding_window" (default, legacy) or "multiscale_bidir"
        (bidirectional multi-scale endpoints — goal pinned to the endpoint the
        actions reach; subsumes augment_reverse and ignores goal_sampling/stride).
      offsets: multiscale endpoint distances, each a multiple of chunk_size.
        Default (8, 16, 24) → per-action frame_step (1, 2, 3) at chunk_size=8.
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
        val_pair_stride: int = 0,
        val_split_level: str = "pair",
        val_names: Sequence[str] | None = None,
        train_exclude_names: Sequence[str] | None = None,
        exclude_objects: Sequence[str] | None = None,
        split: str = "train",
        augment_reverse: bool = False,
        sampling_scheme: str = "sliding_window",
        offsets: Sequence[int] = DEFAULT_MULTISCALE_OFFSETS,
        delta_range: tuple[int, int] = DEFAULT_DELTA_RANGE,
        goal_occupancy_range: tuple[float, float] = DEFAULT_GOAL_OCCUPANCY_RANGE,
        min_goal_visible_frac: float = DEFAULT_MIN_GOAL_VISIBLE_FRAC,
        require_goal_center_on_screen: bool = True,
        min_start_occupancy: float = 1.0,
        max_windows_per_pair: int = 0,
        value_target_mode: str = "cost_to_go",
    ) -> None:
        if goal_sampling not in GOAL_SAMPLING_MODES:
            raise ValueError(f"goal_sampling must be one of {GOAL_SAMPLING_MODES}, got {goal_sampling!r}")
        if sampling_scheme not in SAMPLING_SCHEMES:
            raise ValueError(f"sampling_scheme must be one of {SAMPLING_SCHEMES}, got {sampling_scheme!r}")
        if value_target_mode not in VALUE_TARGET_MODES:
            raise ValueError(f"value_target_mode must be one of {VALUE_TARGET_MODES}, got {value_target_mode!r}")
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if split == "val" and val_pair_stride <= 0 and not val_names:
            raise ValueError("split='val' requires val_pair_stride > 0 or val_names")
        self._val_names = frozenset(val_names) if val_names else None
        # Placements to drop from TRAIN only: the scene/object "crossover" cells a
        # scene-AND-object-disjoint split (scripts/make_val_split.py) must exclude
        # so no val scene or val object ever appears in train.
        self._train_exclude = frozenset(train_exclude_names) if train_exclude_names else None
        # Defective assets, dropped from BOTH splits (pass an empty sequence to keep them).
        self._exclude_objects = (
            DEFAULT_EXCLUDE_OBJECTS if exclude_objects is None else frozenset(exclude_objects)
        )
        self.goal_keys = goal_keys(goal_score_keys)
        self.chunk_size = chunk_size
        self.stride = stride
        self.filter_clamped_goals = filter_clamped_goals
        self.goal_sampling = goal_sampling
        self.augment_reverse = augment_reverse
        self.sampling_scheme = sampling_scheme
        self.offsets = tuple(offsets)
        self.delta_range = (int(delta_range[0]), int(delta_range[1]))
        self.goal_occupancy_range = (float(goal_occupancy_range[0]), float(goal_occupancy_range[1]))
        self.min_goal_visible_frac = float(min_goal_visible_frac)
        self.require_goal_center_on_screen = bool(require_goal_center_on_screen)
        self.min_start_occupancy = float(min_start_occupancy)
        self.max_windows_per_pair = int(max_windows_per_pair)
        self.value_target_mode = value_target_mode
        self._files = list_annotation_files(annotation_roots)
        self._entries: list[_Entry] = []
        for f in self._files:
            if sampling_scheme == "goal_start":
                # Well-framed goal (occupancy in range) + a start delta away; the chunk
                # is the immediate steps toward it. See iter_goal_start_windows.
                windows = iter_goal_start_windows(
                    f,
                    chunk_size=chunk_size,
                    delta_range=self.delta_range,
                    goal_occupancy_range=self.goal_occupancy_range,
                    min_goal_visible_frac=self.min_goal_visible_frac,
                    require_goal_center_on_screen=self.require_goal_center_on_screen,
                    min_start_occupancy=self.min_start_occupancy,
                    max_per_pair=self.max_windows_per_pair,
                )
            elif sampling_scheme == "multiscale_bidir":
                # Bidirectional multi-scale endpoints: goal = the endpoint the actions
                # reach, so the same start with different endpoints forces the action
                # to depend on the goal. Negative offsets already give dolly-OUT, so
                # augment_reverse / goal_sampling are not used in this scheme.
                windows = iter_multiscale_windows(f, chunk_size=chunk_size, offsets=self.offsets)
            else:
                windows = iter_windows(f, chunk_size=chunk_size, stride=stride)
                if augment_reverse:
                    # Reversed trajectories give the dolly-OUT direction the data lacks
                    # (~81% of forward trajectories dolly in); each forward window gets a
                    # reversed counterpart, balancing the action direction ~50/50.
                    windows = itertools.chain(
                        windows, iter_windows(f, chunk_size=chunk_size, stride=stride, reverse=True))
            for window in windows:
                # Defective assets are dropped from every split (see DEFAULT_EXCLUDE_OBJECTS).
                if self._exclude_objects and _window_object(window) in self._exclude_objects:
                    continue
                # Drop scene/object crossover placements from train (leak-free split).
                if (split == "train" and self._train_exclude is not None
                        and window.annotation_path.parent.name in self._train_exclude):
                    continue
                if val_pair_stride > 0 or self._val_names is not None:
                    is_val = _is_val_pair(
                        window.annotation_path, window.pair_idx, val_pair_stride,
                        val_split_level, self._val_names,
                    )
                    if is_val != (split == "val"):
                        continue
                # multiscale_bidir pins the goal to the window's endpoint; sliding_window
                # uses the HER pool (end frame, or [end..last] under uniform_future).
                if sampling_scheme == "goal_start":
                    pool = [window.goal_frame] if window.goal_frame is not None else []
                elif sampling_scheme == "multiscale_bidir" or goal_sampling == "end":
                    pool = [window.end]
                else:
                    pool = [window.end, *window.future]
                candidates: list[tuple[ViewRecord, np.ndarray]] = []
                for view in pool:
                    g = goal_vector(view.raw, self.goal_keys, object_key=_window_object(window))
                    if not np.isfinite(g).all():
                        continue
                    if self.filter_clamped_goals and _is_clamped(view.raw):
                        continue
                    candidates.append((view, g))
                if not candidates:
                    continue
                self._entries.append(_Entry(window, candidates))
                if max_samples and len(self._entries) >= max_samples:
                    return
            if max_samples and len(self._entries) >= max_samples:
                return

    def __len__(self) -> int:
        return len(self._entries)

    def goal_bearings(self) -> np.ndarray:
        """Per-sample subject-frame bearing of the goal, degrees (NaN if not in the goal keys).

        For `goal_start` each sample has exactly one goal candidate; other schemes report
        the first candidate.
        """
        try:
            i = self.goal_keys.index(SUBJECT_BEARING_KEY)
        except ValueError:
            return np.full(len(self._entries), np.nan, dtype=np.float32)
        return np.array(
            [e.candidates[0][1][i] for e in self._entries], dtype=np.float32
        )

    def bearing_balanced_weights(self) -> np.ndarray:
        """Per-sample weights that even out the 8-way view sectors, for a weighted sampler.

        The raw data is strongly skewed — measured over well-framed goals the sectors run
        from ~25% (right) down to ~1% (left), and FRONT views, the most useful photographic
        goal, are among the rarest. Weighting rather than discarding keeps every sample:
        w_i = 1 / count(sector_i), normalized to mean 1.
        """
        bearings = self.goal_bearings()
        sectors = np.array(
            [sector8(float(b)) if np.isfinite(b) else "?" for b in bearings]
        )
        counts = Counter(sectors.tolist())
        w = np.array([1.0 / counts[s] for s in sectors], dtype=np.float64)
        return (w / w.mean()).astype(np.float32)

    def __getitem__(self, idx: int) -> Sample:
        entry = self._entries[idx]
        window = entry.window
        j = int(np.random.randint(len(entry.candidates))) if len(entry.candidates) > 1 else 0
        goal_view, g = entry.candidates[j]
        # Action chunk + value are derived from the window here (lazily), not at init.
        action_chunk = _compute_action_chunk(window)
        # Value = per-step -(geometric distance from each pre-action keyframe pose to
        # the GOAL pose). Pose + subject geometry, NOT bbox score pixels — exact for
        # every frame, no off-screen sentinel. value[0] is the START→GOAL scalar (the
        # legacy value); the sequence adds the intermediate cost-to-go along the chunk.
        # (achieved_profile / profile_delta modes return a (chunk_size, goal_dim) target.)
        value = _compute_value_sequence(window, goal_view, self.value_target_mode, self.goal_keys)
        return Sample(
            start=window.start,
            end=window.end,
            intermediate=window.intermediate,
            action_chunk=action_chunk,
            goal_vec=g,
            value=value,
            chunk_size=window.chunk_size,
            goal=goal_view,
        )
