"""Iterator over v7 placement annotations.

The v7 dataset (`docs/v7_handoff_jooyeol.md` on branch `v7_data_for_cosmos_policy`)
is the only format this project consumes. Layout:

    outputs/v7_stage2_renders/<scene>__<object>/
    ├── data.json
    ├── renders/pair_<pp>_frame_<ff>.jpg     (K_accepted × 32 JPEGs)
    ├── done.flag                            (Stage 2 complete)
    └── scored.flag                          (Stage 3 complete)

`data.json` carries Stage 1 (placement + accepted_pairs[].trajectory_32f) +
Stage 2 (render_records[][].path_rel/bbox/in_frame) + Stage 3 (render_records[][].scores
with 8 V5 keys per frame).

`iter_windows` slides a `chunk_size`-step window over each
`accepted_pairs[i].trajectory_32f` (length 32). Each yielded `TrajectoryWindow`
holds the start frame, end frame, and intermediate frames between them. The
action chunk + goal vector are computed downstream in `BasePolicyDataset`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class ViewRecord:
    """One rendered frame inside a v7 trajectory."""

    annotation_path: Path
    scene: str
    scene_file: str
    scene_scale: float
    object: str
    object_file: str
    pair_idx: int                       # which accepted_pair this frame belongs to
    frame_idx: int                      # 0..31 within trajectory_32f
    object_position: list[float]        # subject_foot (world frame)
    subject_center: list[float]         # subject bbox center (world frame) — for pose-based value
    subject_height: float               # subject height (m) — for pose-based apparent size
    image: str                          # absolute path to the rendered JPEG
    camera_position: list[float]
    camera_forward: list[float]
    camera_up: list[float]
    azimuth: float | None               # frame.yaw_deg (camera-side, not cam→obj)
    elevation: float | None             # frame.pitch_deg
    render_width: int                   # for pixel→angle conversion in the value metric
    render_height: int
    raw: dict                           # frame dict + injected Stage 3 scores


@dataclass
class TrajectoryWindow:
    """A K-step window from a trajectory: start frame, end frame, K-1 intermediates."""

    annotation_path: Path
    scene: str
    scene_file: str
    object: str
    object_file: str
    pair_idx: int
    start_frame_idx: int
    end_frame_idx: int                  # = start_frame_idx + chunk_size
    chunk_size: int
    start: ViewRecord
    end: ViewRecord
    intermediate: list[ViewRecord]      # length chunk_size - 1
    future: list[ViewRecord] = field(default_factory=list)
    # frames AFTER end on the same trajectory (end_frame_idx+1 .. 31) — the
    # HER-"future" goal candidate pool (goal = any frame in [end, 31]).
    keyframes: list[ViewRecord] = field(default_factory=list)
    # the chunk_size+1 frames the action chunk is encoded BETWEEN. For
    # sliding_window these are consecutive ([start, *intermediate, end]); for
    # multiscale_bidir they are strided ([p, p±s, …, p±chunk·s]).
    frame_step: int = 1                 # frames advanced per action (multiscale: o//chunk)
    direction: int = 1                  # +1 forward, -1 reversed (dolly-out)
    # For goal_start windows: the explicit well-framed goal frame (not an HER pool).
    # The action chunk walks start->goal_frame and clamps; `shoot_column` reads arrival
    # off `keyframes` against this. None for sliding_window / multiscale_bidir.
    goal_frame: ViewRecord | None = None


def load_annotation(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _apply_crop_extent(raw: dict, bbox, height: float) -> None:
    """Which END of the subject the frame cuts, and how much of it survives (V12 port).

    Pure geometry off the unclipped signed projection `bbox_xyxy_full`, so it adds only
    gate quantities and does NOT touch the goal-space keys (object_center / bbox_offset
    stay full-bbox, i.e. our current goal space). `visible_frac` is the crop-side-agnostic
    fraction of the subject's vertical extent inside the frame — the gate `_is_well_framed`
    reads; `top_cut_frac`/`bot_cut_frac`/`head_in_frame` carry the side, for reporting.
    """
    try:
        y0, y1 = float(bbox[1]), float(bbox[3])
    except (TypeError, IndexError, ValueError):
        return
    span = y1 - y0
    if span <= 0:
        return
    raw["head_in_frame"] = bool(y0 >= 0.0)
    raw["top_cut_frac"] = max(0.0, -y0) / span
    raw["bot_cut_frac"] = max(0.0, y1 - height) / span
    raw["visible_frac"] = max(0.0, min(y1, height) - max(y0, 0.0)) / span


def _frame_to_view(
    doc: dict,
    annotation_path: Path,
    placement_dir: Path,
    pair_idx: int,
    frame_idx: int,
    frame: dict,
    render_record: dict | None,
) -> ViewRecord:
    """Convert a `trajectory_32f` frame + its `render_records[i][j]` into a `ViewRecord`.

    If `render_record` is absent (Stage 2/3 not yet run for this frame), we
    synthesize the expected `path_rel` and leave scores out — `goal_vector` will
    then return NaN and the sample will be filtered downstream.
    """
    if render_record is not None:
        image_rel = render_record.get("path_rel", f"renders/pair_{pair_idx:02d}_frame_{frame_idx:02d}.jpg")
        scores = render_record.get("scores") or {}
    else:
        image_rel = f"renders/pair_{pair_idx:02d}_frame_{frame_idx:02d}.jpg"
        scores = {}

    raw = dict(frame)
    raw.update(scores)
    raw["frame_idx"] = frame_idx
    if render_record is not None:
        raw["bbox_xyxy_full"] = render_record.get("bbox_xyxy_full")
        raw["in_frame"] = render_record.get("in_frame")
        raw["occupancy_clipped"] = render_record.get("occupancy_clipped")
        # Derive visible_frac (+ crop side) for the goal-start well-framed gate. Pure
        # geometry — leaves object_center / bbox_offset (our goal space) untouched.
        _apply_crop_extent(raw, raw["bbox_xyxy_full"], doc.get("render_height"))

    return ViewRecord(
        annotation_path=annotation_path,
        scene=str(Path(doc.get("scene_file", "")).parent.name) or doc.get("scene", ""),
        scene_file=doc.get("scene_file", ""),
        scene_scale=float(doc.get("scene_scale", 1.0)),
        object=doc.get("object", "") or doc.get("placement", "").split("__")[-1],
        object_file=doc.get("object_file", ""),
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        object_position=list(doc.get("subject_foot") or [0.0, 0.0, 0.0]),
        subject_center=list(doc.get("subject_center") or doc.get("subject_foot") or [0.0, 0.0, 0.0]),
        subject_height=float(doc.get("subject_height") or 1.7),
        image=str(placement_dir / image_rel),
        camera_position=list(frame["pos"]),
        camera_forward=list(frame["forward"]),
        camera_up=list(frame["up"]),
        azimuth=frame.get("yaw_deg"),
        elevation=frame.get("pitch_deg"),
        render_width=int(doc.get("render_width") or 0),
        render_height=int(doc.get("render_height") or 0),
        raw=raw,
    )


def iter_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    stride: int = 1,
) -> Iterator[TrajectoryWindow]:
    """Slide a chunk_size-step window over each accepted_pair's trajectory_32f."""
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        if len(trajectory) <= chunk_size:
            continue
        recs = render_records[pair_idx] if pair_idx < len(render_records) else []
        recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}

        view_records = [
            _frame_to_view(doc, data_json_path, placement_dir, pair_idx, j, trajectory[j], recs_by_idx.get(j))
            for j in range(len(trajectory))
        ]
        for start_idx in range(0, len(trajectory) - chunk_size, stride):
            end_idx = start_idx + chunk_size
            yield TrajectoryWindow(
                annotation_path=data_json_path,
                scene=view_records[start_idx].scene,
                scene_file=view_records[start_idx].scene_file,
                object=view_records[start_idx].object,
                object_file=view_records[start_idx].object_file,
                pair_idx=pair_idx,
                start_frame_idx=start_idx,
                end_frame_idx=end_idx,
                chunk_size=chunk_size,
                start=view_records[start_idx],
                end=view_records[end_idx],
                intermediate=view_records[start_idx + 1 : end_idx],
                future=view_records[end_idx + 1 :],
                keyframes=view_records[start_idx : end_idx + 1],
                frame_step=1,
                direction=1,
            )


def iter_multiscale_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    offsets: Iterable[int] = (8, 16, 24),
) -> Iterator[TrajectoryWindow]:
    """Bidirectional multi-scale endpoint windows — makes actions depend on the goal.

    For each start frame `p` and each signed offset `±o` (o in `offsets`) whose
    endpoint `p±o` exists, emit a window whose chunk_size actions traverse `p`→`p±o`.
    The endpoint frame IS the goal, so the SAME start with DIFFERENT endpoints yields
    DIFFERENT action targets — forcing goal-conditioning instead of collapse to
    f(state) (the sliding-window failure mode, where the action is pinned to the
    window while the HER goal varies independently). Negative offsets play the path
    backward (dolly-out), subsuming the reverse augmentation.

    Each offset must be a positive multiple of chunk_size; the ratio is the per-action
    `frame_step` s (o=8→1, 16→2, 24→3 at chunk_size=8). The chunk is re-encoded between
    the STRIDED keyframes `[p, p±s, …, p±chunk·s]` downstream — not by summing single
    deltas (which don't compose in the camera-local basis).
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    offset_list = sorted({int(o) for o in offsets})
    for o in offset_list:
        if o <= 0 or o % chunk_size != 0:
            raise ValueError(
                f"offset {o} must be a positive multiple of chunk_size={chunk_size}"
            )

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        n = len(trajectory)
        if n <= 1:
            continue
        recs = render_records[pair_idx] if pair_idx < len(render_records) else []
        recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
        view_records = [
            _frame_to_view(doc, data_json_path, placement_dir, pair_idx, j, trajectory[j], recs_by_idx.get(j))
            for j in range(n)
        ]
        for start_idx in range(n):
            for o in offset_list:
                step = o // chunk_size
                for direction in (1, -1):
                    end_idx = start_idx + direction * o
                    if end_idx < 0 or end_idx >= n:
                        continue
                    keyframes = [view_records[start_idx + direction * step * k] for k in range(chunk_size + 1)]
                    yield TrajectoryWindow(
                        annotation_path=data_json_path,
                        scene=keyframes[0].scene,
                        scene_file=keyframes[0].scene_file,
                        object=keyframes[0].object,
                        object_file=keyframes[0].object_file,
                        pair_idx=pair_idx,
                        start_frame_idx=start_idx,
                        end_frame_idx=end_idx,
                        chunk_size=chunk_size,
                        start=keyframes[0],
                        end=keyframes[-1],
                        intermediate=keyframes[1:-1],
                        future=[],                      # goal pinned to endpoint; no HER pool
                        keyframes=keyframes,
                        frame_step=step,
                        direction=direction,
                    )


DEFAULT_DELTA_RANGE = (0, 32)
DEFAULT_NEAR_FRACTION = 0.25
DEFAULT_GOAL_OCCUPANCY_RANGE = (10.0, 80.0)   # occupancy is 0-100 in our scores
DEFAULT_MIN_GOAL_VISIBLE_FRAC = 0.35


def shoot_column(window: TrajectoryWindow) -> np.ndarray:
    """(chunk_size,) latched 0/1 `shoot` channel: 0 before goal arrival, 1 from it on.

    Verbatim port of V12 `dataset_base.shoot_column`. Arrival is read off `keyframes`
    (the clamped walk), NOT a signed index diff — a goal can sit before the start
    (delta = |g - s|), and a signed diff would then latch the whole chunk. Pairs with a
    zero pose action past arrival (the walk clamps at the goal): "you are there, hold
    still, take the photo." Falls back to the last keyframe as goal when goal_frame is
    unset (non-goal_start windows), where it is all-zero by construction.
    """
    frames = window.keyframes if window.keyframes else [
        window.start, *window.intermediate, window.end]
    goal_idx = window.goal_frame.frame_idx if window.goal_frame is not None else frames[-1].frame_idx
    k = next((i for i, f in enumerate(frames) if f.frame_idx == goal_idx), len(frames))
    return (np.arange(window.chunk_size) >= k).astype(np.float32)


def _is_well_framed(
    raw: dict, min_visible_frac: float, width: float, height: float,
    require_center_on_screen: bool = True,
) -> bool:
    """Composition gate a goal frame must pass on top of its occupancy band (V12 port).

    Gates on `visible_frac` (fraction of the subject's vertical extent inside the frame,
    from `_apply_crop_extent`); falls back to the AREA ratio `body_in_frame_ratio` only
    when visible_frac is absent. Optionally also requires the subject centre on screen.
    """
    if not isinstance(raw, dict):
        return False
    if min_visible_frac > 0.0:
        vis = raw.get("visible_frac")
        if vis is None:
            body = raw.get("body_in_frame_ratio")
            if body is None or float(body) / 100.0 < min_visible_frac:
                return False
        elif float(vis) < min_visible_frac:
            return False
    if require_center_on_screen:
        cx, cy = raw.get("object_center_x"), raw.get("object_center_y")
        if cx is None or cy is None:
            return False
        if not (0.0 <= float(cx) <= width and 0.0 <= float(cy) <= height):
            return False
    return True


def iter_goal_start_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    delta_range: tuple[int, int] = DEFAULT_DELTA_RANGE,
    near_fraction: float = DEFAULT_NEAR_FRACTION,
    goal_occupancy_range: tuple[float, float] = DEFAULT_GOAL_OCCUPANCY_RANGE,
    min_goal_visible_frac: float = DEFAULT_MIN_GOAL_VISIBLE_FRAC,
    require_goal_center_on_screen: bool = True,
    min_start_occupancy: float = 1.0,
    max_per_pair: int = 0,
    seed: int = 0,
) -> Iterator[TrajectoryWindow]:
    """Well-framed GOAL + a start some distance away; the chunk is the IMMEDIATE steps
    toward it, clamped at the goal (V12 port of `iter_goal_start_windows`).

    Per pair: a goal `g` is any frame whose occupancy is in `goal_occupancy_range` and
    that is well-framed (`_is_well_framed`); a start `s` is any frame with
    `delta_range[0] <= |g - s| <= delta_range[1]` and occupancy > `min_start_occupancy`.
    The action chunk is the `chunk_size` steps from s toward g, one trajectory frame
    each, CLAMPED at g — so deltas past arrival are zero (the `shoot=1` supervision).
    Both signs of (g - s) are used (dolly-out for free). `max_per_pair` caps kept pairs
    (deterministic, seeded), stratified so near (|g-s| < chunk_size) windows keep a
    `near_fraction` share — else the pool is dominated by "already there, hold still".
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []
    width = float(doc.get("render_width") or 0.0)
    height = float(doc.get("render_height") or 0.0)

    d_min, d_max = int(delta_range[0]), int(delta_range[1])
    if d_min < 0 or d_max < d_min:
        raise ValueError(f"bad delta_range {delta_range}")
    occ_lo, occ_hi = float(goal_occupancy_range[0]), float(goal_occupancy_range[1])

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        n = len(trajectory)
        if n <= chunk_size:
            continue
        recs = render_records[pair_idx] if pair_idx < len(render_records) else []
        recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
        view_records = [
            _frame_to_view(doc, data_json_path, placement_dir, pair_idx, j, trajectory[j], recs_by_idx.get(j))
            for j in range(n)
        ]
        occupancy = [v.raw.get("occupancy") for v in view_records]

        pairs_sg: list[tuple[int, int]] = []
        for g in range(n):
            og = occupancy[g]
            if og is None or not (occ_lo <= float(og) <= occ_hi):
                continue
            if not _is_well_framed(view_records[g].raw, min_goal_visible_frac, width, height,
                                   require_goal_center_on_screen):
                continue
            for s in range(n):
                delta = abs(g - s)
                if delta < d_min or delta > d_max:
                    continue
                os_ = occupancy[s]
                if os_ is None or float(os_) <= min_start_occupancy:
                    continue
                pairs_sg.append((s, g))

        if max_per_pair and len(pairs_sg) > max_per_pair:
            rng = random.Random(f"{placement_dir.name}:{pair_idx}:{seed}")
            # Stratify by distance: near starts (|g-s| < chunk_size) vastly outnumber
            # far ones, so a flat draw would make most data "already there, hold still".
            near = [sg for sg in pairs_sg if abs(sg[1] - sg[0]) < chunk_size]
            far = [sg for sg in pairs_sg if abs(sg[1] - sg[0]) >= chunk_size]
            n_near = min(len(near), int(round(max_per_pair * float(near_fraction))))
            n_far = min(len(far), max_per_pair - n_near)
            n_near = min(len(near), max_per_pair - n_far)       # backfill if far is short
            picked = rng.sample(near, n_near) + rng.sample(far, n_far)
            pairs_sg = sorted(picked)

        for s, g in pairs_sg:
            direction = 1 if g > s else -1
            # Walk toward g one frame at a time and CLAMP there: past-arrival keyframes
            # repeat the goal, so their pose deltas are zero and shoot latches to 1.
            idx = [s + direction * k for k in range(chunk_size + 1)]
            idx = [min(i, g) if direction > 0 else max(i, g) for i in idx]
            keyframes = [view_records[i] for i in idx]
            yield TrajectoryWindow(
                annotation_path=data_json_path,
                scene=keyframes[0].scene,
                scene_file=keyframes[0].scene_file,
                object=keyframes[0].object,
                object_file=keyframes[0].object_file,
                pair_idx=pair_idx,
                start_frame_idx=keyframes[0].frame_idx,
                end_frame_idx=keyframes[-1].frame_idx,
                chunk_size=chunk_size,
                start=keyframes[0],
                end=keyframes[-1],
                intermediate=keyframes[1:-1],
                future=[],                          # goal is explicit, not an HER pool
                keyframes=keyframes,
                frame_step=1,
                direction=direction,
                goal_frame=view_records[g],
            )


def load_val_names(spec) -> list[str] | None:
    """Resolve a `val_names` config value into a list of scene names.

    Accepts:
      - a list (inline in YAML) -> returned as-is;
      - a path to `.yaml`/`.yml` -> the manifest's `scenes[].name` (new format,
        see scripts/make_val_split.py), or a bare top-level list;
      - a path to `.json` -> a list, or `{"scenes": [...]}`;
      - a path to `.txt` -> one name per line, `#` comments allowed (legacy).
    Returns None for a falsy spec.
    """
    if not spec:
        return None
    if isinstance(spec, (list, tuple)):
        return list(spec)
    path = Path(spec)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and "scenes" in doc:
            return [s["name"] if isinstance(s, dict) else s for s in doc["scenes"]]
        return list(doc)
    if suffix == ".json":
        doc = json.loads(path.read_text())
        return doc["scenes"] if isinstance(doc, dict) and "scenes" in doc else list(doc)
    # legacy .txt: one name per line, '#' comments
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def list_annotation_files(roots: Iterable[str | Path]) -> list[Path]:
    """Find every `data.json` under each root.

    Each v7 placement contributes exactly one `data.json`, so we glob for that name.
    """
    out: list[Path] = []
    for r in roots:
        rp = Path(r)
        if rp.is_file() and rp.name == "data.json":
            out.append(rp)
        elif rp.is_dir():
            out.extend(sorted(rp.glob("**/data.json")))
    return out


__all__ = [
    "ViewRecord",
    "TrajectoryWindow",
    "iter_windows",
    "iter_multiscale_windows",
    "iter_goal_start_windows",
    "shoot_column",
    "list_annotation_files",
    "load_val_names",
    "load_annotation",
]
