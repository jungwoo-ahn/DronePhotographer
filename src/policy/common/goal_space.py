"""Shot-profile goal space utilities.

The goal vector for the policy is a subset of the V5 score keys defined in
`src/scoring/bbox_control.py`. Annotations may or may not have all keys present
(the v6 placement JSONs include camera azimuth/elevation but not the full V5
detection-derived scores yet), so the loader tolerates missing keys by filling
NaN — callers decide whether to skip or mask those samples downstream.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.scoring import V5_SCORE_KEYS

DEFAULT_GOAL_KEYS: list[str] = list(V5_SCORE_KEYS)

# v7 render resolution (scripts/v7_stage2_render.py default --resolution 1024 768).
# The pixel-valued score keys are normalized against these. Override if the
# render resolution changes.
RENDER_WIDTH: float = 1024.0
RENDER_HEIGHT: float = 768.0

# Per-key value ranges used to map raw V5 scores to [-1, 1]. Calibrated against
# the actual integer schema emitted by `src.scoring.compute_v5_scores`:
#   - occupancy, body_in_frame_ratio : 0-100 percent
#   - cam_to_obj_azimuth_deg         : 0-360 (NOTE: cyclic — see normalize_goal)
#   - cam_to_obj_elevation_deg       : -90..90 (v2 convention, cam above -> negative)
#   - object_center_x / _y           : bbox-center pixel coords, (0, W) / (0, H)
#   - bbox_x_offset / _y_offset      : HALF the bbox width/height in pixels (>=0),
#                                      a subject-size cue, normalized against (0, W)/(0, H)
DEFAULT_V5_RANGES: dict[str, tuple[float, float]] = {
    "occupancy": (0.0, 100.0),
    "body_in_frame_ratio": (0.0, 100.0),
    "cam_to_obj_azimuth_deg": (0.0, 360.0),
    "cam_to_obj_elevation_deg": (-90.0, 90.0),
    "object_center_x": (0.0, RENDER_WIDTH),
    "object_center_y": (0.0, RENDER_HEIGHT),
    "bbox_x_offset": (0.0, RENDER_WIDTH),
    "bbox_y_offset": (0.0, RENDER_HEIGHT),
}

# Score keys that are angles in degrees and wrap around (cyclic). Linear
# normalization of these has a discontinuity at the wrap point (0 deg == 360 deg
# map to opposite ends of [-1, 1]). For the prototype we accept the seam; a
# sin/cos encoding (which would add one dimension per cyclic key) is the proper
# fix if azimuth conditioning proves unreliable near 0/360.
CYCLIC_GOAL_KEYS: frozenset[str] = frozenset({"cam_to_obj_azimuth_deg"})


def goal_keys(custom: Sequence[str] | None = None) -> list[str]:
    return list(custom) if custom else list(DEFAULT_GOAL_KEYS)


def derive_partial_v5_from_final_image(final_image: Mapping) -> dict[str, float]:
    """Best-effort extraction of V5 keys from a v6 `final_images` entry.

    v6 placement JSONs store `azimuth` and `elevation` per rendered view but not
    the full detection-derived V5 scores. Map what's there; leave the rest absent.
    """
    out: dict[str, float] = {}
    if "azimuth" in final_image:
        out["cam_to_obj_azimuth_deg"] = float(final_image["azimuth"])
    if "elevation" in final_image:
        out["cam_to_obj_elevation_deg"] = float(final_image["elevation"])
    return out


def goal_vector(
    annotation_view: Mapping,
    keys: Sequence[str] | None = None,
    *,
    score_prefix: str = "score_",
) -> np.ndarray:
    """Extract a goal vector from an annotation.

    Looks for each key under the bare name first (e.g. `cam_to_obj_azimuth_deg`),
    then under `score_<key>` (the convention written by `score_annotations.py`),
    then under the inline mapping returned by `derive_partial_v5_from_final_image`.
    Missing keys yield NaN.
    """
    keys = goal_keys(keys)
    derived = derive_partial_v5_from_final_image(annotation_view)
    out = np.full(len(keys), np.nan, dtype=np.float32)
    for i, k in enumerate(keys):
        if k in annotation_view:
            out[i] = float(annotation_view[k])
        elif (sk := f"{score_prefix}{k}") in annotation_view:
            out[i] = float(annotation_view[sk])
        elif k in derived:
            out[i] = derived[k]
    return out


def normalize_goal(
    vec: np.ndarray,
    keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Map per-key values to [-1, 1] using `ranges` (defaults to DEFAULT_V5_RANGES).

    Out-of-range values are clipped. NaN passes through.
    """
    keys = goal_keys(keys)
    ranges = ranges or DEFAULT_V5_RANGES
    out = np.empty_like(vec, dtype=np.float32)
    for i, k in enumerate(keys):
        lo, hi = ranges.get(k, (0.0, 1.0))
        span = hi - lo
        if span <= 0:
            out[i] = vec[i]
        else:
            out[i] = 2.0 * (vec[i] - lo) / span - 1.0
            if not np.isnan(out[i]):
                out[i] = float(np.clip(out[i], -1.0, 1.0))
    return out


def denormalize_goal(
    vec: np.ndarray,
    keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    keys = goal_keys(keys)
    ranges = ranges or DEFAULT_V5_RANGES
    out = np.empty_like(vec, dtype=np.float32)
    for i, k in enumerate(keys):
        lo, hi = ranges.get(k, (0.0, 1.0))
        out[i] = 0.5 * (vec[i] + 1.0) * (hi - lo) + lo
    return out


def has_all_keys(annotation_view: Mapping, keys: Iterable[str] | None = None) -> bool:
    keys = goal_keys(list(keys) if keys else None)
    vec = goal_vector(annotation_view, keys)
    return bool(np.isfinite(vec).all())
