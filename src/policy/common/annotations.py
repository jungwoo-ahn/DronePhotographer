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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from src.policy.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH
from src.scoring.bbox_control import compute_v5_scores


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
    # the chunk_size+1 frames whose consecutive pose deltas ARE the action chunk
    # (and whose per-frame poses feed the per-step value): [start, …, end].
    # sliding_window → contiguous start..end; multiscale_bidir → strided by
    # frame_step. `_compute_action_chunk` / the value sequence read this list.
    frame_step: int = 1                 # real trajectory frames spanned per action (1|2|3)
    direction: int = 1                  # +1 forward along the trajectory, -1 reversed


def load_annotation(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_clamped_scores(raw: dict) -> bool:
    """The v7 stage-3 near-plane sentinel: occupancy 0 with a zero bbox size."""
    return raw.get("occupancy") == 0 and raw.get("bbox_y_offset") == 0


def _recover_clamped_goal(raw: dict) -> None:
    """Recover near-plane-clamped goal frames in place (no re-render).

    When the dolly closes in, the mesh-tight bbox blows up and the v7 stage-3
    scorer baked a zeroed sentinel (occupancy 0, bbox keys 0; az/el kept). Those
    frames are valid extreme close-ups — `occupancy_clipped`~1.0, `in_frame`
    True — that the old filter dropped. Recompute a bounded profile from the
    stored full bbox via the (now non-zeroing) `compute_v5_scores`, and take
    occupancy from the stored `occupancy_clipped` ground truth. No-op for
    non-sentinel frames and for genuine no-projection frames (bbox is None).
    """
    if not _is_clamped_scores(raw):
        return
    bbox = raw.get("bbox_xyxy_full")
    if not bbox:
        return  # genuine no-projection (subject behind camera) — leave zeroed
    az = float(raw.get("cam_to_obj_azimuth_deg") or 0.0)
    el = float(raw.get("cam_to_obj_elevation_deg") or 0.0)
    fixed = compute_v5_scores(int(RENDER_WIDTH), int(RENDER_HEIGHT), [float(v) for v in bbox], az, el)
    occ_clip = raw.get("occupancy_clipped")
    if occ_clip is not None:
        fixed["occupancy"] = max(0, min(100, int(round(100.0 * float(occ_clip)))))
    raw.update(fixed)


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
        _recover_clamped_goal(raw)

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


def _pair_views(
    doc: dict,
    data_json_path: Path,
    placement_dir: Path,
    pair_idx: int,
    trajectory: list,
    render_records: list,
) -> list[ViewRecord]:
    """Build the per-frame `ViewRecord` list for one accepted_pair's trajectory."""
    recs = render_records[pair_idx] if pair_idx < len(render_records) else []
    recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
    return [
        _frame_to_view(doc, data_json_path, placement_dir, pair_idx, j, trajectory[j], recs_by_idx.get(j))
        for j in range(len(trajectory))
    ]


def iter_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    stride: int = 1,
    reverse: bool = False,
) -> Iterator[TrajectoryWindow]:
    """Slide a chunk_size-step window over each accepted_pair's trajectory_32f.

    reverse=True plays each trajectory backward (camera path end->start): the action
    chunks become the inverse motion (dolly-OUT where the forward path dollies in) and
    the HER-"future" goals are FARTHER frames. Used to balance the dataset's strong
    far->near bias (~81% of trajectories dolly in). No re-render — the same rendered
    frames are reused in reverse order.
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        if len(trajectory) <= chunk_size:
            continue
        view_records = _pair_views(doc, data_json_path, placement_dir, pair_idx, trajectory, render_records)
        if reverse:
            view_records = list(reversed(view_records))
        n = len(view_records)
        for start_idx in range(0, n - chunk_size, stride):
            end_idx = start_idx + chunk_size
            yield TrajectoryWindow(
                annotation_path=data_json_path,
                scene=view_records[start_idx].scene,
                scene_file=view_records[start_idx].scene_file,
                object=view_records[start_idx].object,
                object_file=view_records[start_idx].object_file,
                pair_idx=pair_idx,
                start_frame_idx=view_records[start_idx].frame_idx,
                end_frame_idx=view_records[end_idx].frame_idx,
                chunk_size=chunk_size,
                start=view_records[start_idx],
                end=view_records[end_idx],
                intermediate=view_records[start_idx + 1 : end_idx],
                future=view_records[end_idx + 1 :],
                keyframes=view_records[start_idx : end_idx + 1],
                frame_step=1,
                direction=-1 if reverse else 1,
            )


def iter_multiscale_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    offsets: Iterable[int] = (8, 16, 24),
) -> Iterator[TrajectoryWindow]:
    """Bidirectional multi-scale endpoint windows — makes actions depend on the goal.

    For each start frame `p` and each signed offset `±o` (o in `offsets`) whose
    endpoint `p±o` exists in the trajectory, emit a window whose `chunk_size`
    actions traverse from `p` to that endpoint. The endpoint frame IS the goal
    (its profile), so the SAME start with DIFFERENT endpoints yields DIFFERENT
    action targets — forcing the policy to condition on the goal instead of
    collapsing to `f(state)` (the failure mode of the sliding-window scheme, where
    the action is pinned to the window while the HER goal varies independently).

    Each offset must be a positive multiple of `chunk_size`; the ratio is the
    per-action `frame_step` s (o=8→s=1, 16→2, 24→3 at chunk_size=8). The action
    chunk is re-encoded between the STRIDED keyframes `[p, p±s, …, p±(chunk_size·s)]`
    downstream — NOT by summing single-step deltas, which do not compose in the
    camera-local basis. Negative offsets play the path backward (dolly-OUT), so
    this subsumes the `reverse=` augmentation of `iter_windows`.

    On a 32-frame trajectory with offsets (8,16,24) this yields 96 windows/pair
    (24+16+8 forward + 24+16+8 reverse), before any downstream goal filtering.
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
                f"offset {o} must be a positive multiple of chunk_size={chunk_size} "
                "(so it maps to an integer per-action frame_step)"
            )

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        n = len(trajectory)
        if n <= 1:
            continue
        view_records = _pair_views(doc, data_json_path, placement_dir, pair_idx, trajectory, render_records)
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
                        start_frame_idx=keyframes[0].frame_idx,
                        end_frame_idx=keyframes[-1].frame_idx,
                        chunk_size=chunk_size,
                        start=keyframes[0],
                        end=keyframes[-1],
                        intermediate=keyframes[1:-1],
                        future=[],                      # goal is pinned to the endpoint; no HER pool
                        keyframes=keyframes,
                        frame_step=step,
                        direction=direction,
                    )


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
    "list_annotation_files",
    "load_annotation",
]
