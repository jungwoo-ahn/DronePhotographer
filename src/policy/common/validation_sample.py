"""Resolve a v7 validation sample into everything needed to set up + re-render it.

A v7 placement's `data.json` carries the start poses + render settings but NOT the
scene/object transform the render used (`scene_scale`, object `rotation`/`scale`).
Those live in the upstream **VLM v6 placement record**
(`data/vlm_object_placing_v6_*/<placement>.json`: top-level `scene_scale`, and
`placements[accepted].{position, rotation, scale}`). This module joins the two into
a `ValidationSample` and emits an (extended) `run_info` the Blender setup worker
consumes, so re-renders match the dataset (scene scaled, object placed with the
right transform).

This is the data-resolution half of issue #23 points 1-2 ("setup/load scene+object
given a validation sample"); the Blender-side application lives in the setup render
worker. Pure-Python and unit-tested against the real records.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# v7 camera intrinsics (scripts/v7_stage2_render.py defaults; DJI-Mini-class lens).
V7_FOCAL_LENGTH_MM = 24.0
V7_SENSOR_WIDTH_MM = 12.8
V7_SENSOR_HEIGHT_MM = 9.6
# v7 Cycles render options (constants in the v7 configure_renderer).
V7_RENDER_OPTIONS = {
    "max_bounces": 4, "diffuse_bounces": 2, "glossy_bounces": 2,
    "transmission_bounces": 2, "volume_bounces": 0, "transparent_max_bounces": 4,
    "sky_strength": 0.1, "persistent_data": False,
}


@dataclass
class ValidationSample:
    """A v7 placement resolved for re-rendering (scene + object + poses + intrinsics)."""

    placement: str
    scene_file: str
    object_file: str
    scene_scale: float
    object_position: list[float]        # where the object is placed (v6 accepted position)
    object_rotation_xyz: list[float]    # radians (full euler)
    object_scale: float
    subject_center: list[float]         # object center, for the az/el pose proxy
    render_width: int
    render_height: int
    render_samples: int
    start_poses: list[dict] = field(default_factory=list)   # accepted_pairs[].start
    focal_length: float = V7_FOCAL_LENGTH_MM
    sensor_width: float = V7_SENSOR_WIDTH_MM
    sensor_height: float = V7_SENSOR_HEIGHT_MM
    render_device: str = "GPU"   # faithful re-render uses GPU like the dataset (OPTIX/CUDA);
                                 # BlenderDrone._configure_gpu falls back to CPU if no GPU is found.

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_width / (2.0 * self.focal_length)))

    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_height / (2.0 * self.focal_length)))

    def start_pose(self, pair_idx: int = 0) -> tuple[list[float], list[float], list[float]]:
        """(position, forward, up) of the pair's start frame (world frame)."""
        s = self.start_poses[pair_idx]
        return s["pos"], s["forward"], s["up"]

    def to_run_info(self) -> dict:
        """Extended run_info for the Blender setup worker.

        Adds `scene_scale` and full-euler `rotation_xyz_rad` to the stock BlenderDrone
        schema — the setup worker applies `apply_scene_scale` and places the object
        with the v6 transform (the stock drone only handles `rotation_z_deg` and no
        scene scale).
        """
        return {
            "input_scene": self.scene_file,
            "input_object": self.object_file,
            "scene_scale": self.scene_scale,
            "scale": self.object_scale,
            "rotation_xyz_rad": list(self.object_rotation_xyz),
            "options": {
                "object_position": list(self.object_position),
                "resolution": [self.render_width, self.render_height],
                "focal_length": self.focal_length,
                "sensor_width": self.sensor_width,
                "sensor_height": self.sensor_height,
                **V7_RENDER_OPTIONS,
                "samples": self.render_samples,
                "render_device": self.render_device,   # last → wins over V7_RENDER_OPTIONS
            },
        }


def _accepted_placement(vlm_doc: dict) -> dict:
    """The accepted candidate from a VLM v6 record (else the first)."""
    placements = vlm_doc.get("placements") or []
    if not placements:
        raise ValueError("VLM v6 record has no placements")
    for p in placements:
        if p.get("accepted"):
            return p
    return placements[0]


def load_validation_sample(
    data_json_path: str | Path,
    vlm_placements_dir: str | Path,
    *,
    require_vlm: bool = True,
) -> ValidationSample:
    """Join a v7 `data.json` with its VLM v6 record into a `ValidationSample`.

    `vlm_placements_dir` is the dir of per-placement VLM v6 jsons
    (`data/vlm_object_placing_v6_*/`). Raises if the matching record is missing and
    `require_vlm` (the transform params can't be faithfully recovered otherwise).
    """
    data = json.loads(Path(data_json_path).read_text())
    placement = data["placement"]
    vlm_path = Path(vlm_placements_dir) / f"{placement}.json"

    if vlm_path.exists():
        vlm = json.loads(vlm_path.read_text())
        scene_scale = float(vlm["scene_scale"])
        acc = _accepted_placement(vlm)
        object_position = [float(v) for v in acc["position"]]
        object_rotation = [float(v) for v in acc.get("rotation", [0.0, 0.0, 0.0])]
        object_scale = float(acc.get("scale", 1.0))
    elif require_vlm:
        raise FileNotFoundError(
            f"VLM v6 placement record not found for '{placement}' in {vlm_placements_dir}. "
            "It holds scene_scale / object rotation / object scale, which data.json lacks. "
            "Pass require_vlm=False to fall back to identity transforms (NOT render-faithful)."
        )
    else:  # explicit, non-faithful fallback
        scene_scale = 1.0
        object_position = [float(v) for v in data.get("subject_foot", [0.0, 0.0, 0.0])]
        object_rotation = [0.0, 0.0, 0.0]
        object_scale = 1.0

    return ValidationSample(
        placement=placement,
        scene_file=data["scene_file"],
        object_file=data["object_file"],
        scene_scale=scene_scale,
        object_position=object_position,
        object_rotation_xyz=object_rotation,
        object_scale=object_scale,
        subject_center=[float(v) for v in data.get("subject_center", object_position)],
        render_width=int(data.get("render_width", 1024)),
        render_height=int(data.get("render_height", 768)),
        render_samples=int(data.get("render_samples", 32)),
        start_poses=[p["start"] for p in data.get("accepted_pairs", []) if "start" in p],
    )


__all__ = ["ValidationSample", "load_validation_sample", "V7_FOCAL_LENGTH_MM",
           "V7_SENSOR_WIDTH_MM", "V7_SENSOR_HEIGHT_MM", "V7_RENDER_OPTIONS"]
