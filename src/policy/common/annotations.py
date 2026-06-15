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


def load_annotation(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


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
            )


def list_annotation_files(roots: Iterable[str | Path]) -> list[Path]:
    """Find every `data.json` under each root.

    Each v7 placement contributes exactly one `data.json`, so we glob for that name.
    """
    import os

    out: list[Path] = []
    for r in roots:
        rp = Path(r)
        if rp.is_file() and rp.name == "data.json":
            out.append(rp)
        elif rp.is_dir():
            # `rp.glob("**/data.json")` does NOT recurse into symlinked
            # subdirectories on Python < 3.13, and `data/trajectories/`
            # is canonically a tree of symlinks per
            # `data/trajectories/README.md`. Use os.walk with followlinks=True.
            for root, _, files in os.walk(rp, followlinks=True):
                if "data.json" in files:
                    out.append(Path(root) / "data.json")
    out.sort()
    return out


__all__ = [
    "ViewRecord",
    "TrajectoryWindow",
    "iter_windows",
    "list_annotation_files",
    "load_annotation",
]
