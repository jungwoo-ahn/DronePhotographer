"""Per-asset subject facing: world-frame azimuth -> SUBJECT-frame bearing (V12 port).

The stored `cam_to_obj_azimuth_deg` is a WORLD-frame angle. As a goal it is ambiguous:
the same number means "facing the camera" for one asset and "back turned" for another.
Each asset has a baked canonical facing (turntable renders + face detection + human
verification) in `configs/policy/facing_map_final.json`:

    {"<object key>": {"front_az": <deg>, "facing_world_deg": <deg>}, ...}

The subject-frame bearing is  `(front_az - azimuth) mod 360`, so 0 = seen from the
front, 90 = subject's RIGHT, 180 = behind, 270 = subject's LEFT — scene/asset-agnostic
and readable straight off the image, which is what makes it usable as a goal.

Used only by the VLA natural-language goal path (`goal_text.goal_prompt`); DP/cosmos keep
the raw world-azimuth goal space.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FACING_MAP_PATH = _REPO_ROOT / "configs" / "policy" / "facing_map_final.json"

SECTOR8 = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)


def sector8(bearing_deg: float) -> str:
    """8-way view word for a subject-frame bearing (45-deg bins centred on the labels)."""
    return SECTOR8[int(((float(bearing_deg) + 22.5) % 360.0) // 45.0)]


@lru_cache(maxsize=8)
def load_facing_map(path: str | Path | None = None) -> Mapping[str, dict]:
    """Load (and cache) the per-object facing map."""
    p = Path(path) if path is not None else DEFAULT_FACING_MAP_PATH
    with open(p) as fh:
        return json.load(fh)


def front_azimuth(object_key: str, path: str | Path | None = None) -> float | None:
    """Camera azimuth (deg) from which `object_key`'s front is seen, or None if unmapped."""
    entry = load_facing_map(path).get(object_key)
    if not entry or entry.get("front_az") is None:
        return None
    return float(entry["front_az"])


def subject_bearing_deg(
    world_azimuth_deg: float, object_key: str, path: str | Path | None = None,
    *, yaw_deg: float = 0.0,
) -> float | None:
    """World `cam_to_obj_azimuth_deg` -> subject-frame bearing in [0, 360), or None if the
    object has no facing entry. `yaw_deg` is the placement's subject rotation
    (`placement_yaw_deg`, 0 on original renders): effective_front = front_az + yaw."""
    front = front_azimuth(object_key, path)
    if front is None or world_azimuth_deg is None or not math.isfinite(float(world_azimuth_deg)):
        return None
    return (front + float(yaw_deg) - float(world_azimuth_deg)) % 360.0


__all__ = ["DEFAULT_FACING_MAP_PATH", "SECTOR8", "sector8", "load_facing_map",
           "front_azimuth", "subject_bearing_deg"]
