"""Profile-distance reward / value target.

Per issue #18, the value head's target is a **distance between the achieved
profile and the goal profile**, exposed as a reward (negative distance).

The recommended distance is `geometric_profile_distance` (used for the value
target): it maps each profile back to the ~5 geometric DOF of the camera-subject
configuration and measures a clean, weight-free product metric there —

    d² = d_view²  (great-circle angle on the viewing sphere; handles azimuth
                   cyclicity + polar degeneracy)
       + d_size²  (Δ apparent angular subtense of the subject)
       + d_aim²   (Δ optical-axis offset, i.e. where the subject sits off-axis)

All three terms are angles (radians) in the same camera-subject geometry, so
they're summed with equal weight — there are no per-key weights to tune. `atan`
on the size/aim pixel quantities makes off-frame / huge-bbox values saturate
gracefully (→ ±π/2) instead of blowing up. The metric is scale-invariant, which
matches the profile's scene-agnostic design.

`score_distance` (the older weighted-L2 over normalized keys) is kept for the
eval pose-proxy and ablations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.policy.common.goal_space import goal_keys, normalize_goal


# --- Geometric profile distance (value target) --------------------------------


@dataclass
class CameraIntrinsics:
    """Pixel focal lengths, used to turn pixel scores into angles.

    Defaults match the v7 render (`scripts/v7_stage2_render.py`): a 24 mm lens on
    a 12.8 × 9.6 mm sensor. `focal_px = focal_mm / sensor_mm * pixels`.
    """

    width: int
    height: int
    focal_px_x: float
    focal_px_y: float

    @classmethod
    def from_render(
        cls,
        width: int,
        height: int,
        focal_mm: float = 24.0,
        sensor_w_mm: float = 12.8,
        sensor_h_mm: float = 9.6,
    ) -> "CameraIntrinsics":
        return cls(
            width=int(width),
            height=int(height),
            focal_px_x=focal_mm / sensor_w_mm * width,
            focal_px_y=focal_mm / sensor_h_mm * height,
        )


def profile_to_geometry(scores: Mapping[str, float], intr: CameraIntrinsics) -> dict[str, float]:
    """Recover the ~5 geometric DOF (radians) from an 8-key V5 profile.

    Returns:
      az, el        : viewing direction around the subject (from the cam_to_obj angles)
      size          : subject half angular subtense (atan of the bbox half-height)
      aim_x, aim_y  : optical-axis offset — angle of the subject's image-center
                      off the principal axis (atan of object_center − frame_center)
    """
    az = math.radians(float(scores.get("cam_to_obj_azimuth_deg", 0.0)))
    el = math.radians(float(scores.get("cam_to_obj_elevation_deg", 0.0)))
    # apparent size: half-vertical subtense; atan bounds off-frame blow-ups to <pi/2
    size = math.atan(float(scores.get("bbox_y_offset", 0.0)) / intr.focal_px_y)
    aim_x = math.atan((float(scores.get("object_center_x", intr.width / 2)) - intr.width / 2) / intr.focal_px_x)
    aim_y = math.atan((float(scores.get("object_center_y", intr.height / 2)) - intr.height / 2) / intr.focal_px_y)
    return {"az": az, "el": el, "size": size, "aim_x": aim_x, "aim_y": aim_y}


def _great_circle(az_a: float, el_a: float, az_b: float, el_b: float) -> float:
    """Great-circle angle (radians) between two viewing directions on the sphere.

    Uses the spherical law of cosines. Naturally handles azimuth wraparound (via
    cos of the azimuth difference) and polar degeneracy (azimuth matters less as
    elevation → ±90°)."""
    cos_d = math.sin(el_a) * math.sin(el_b) + math.cos(el_a) * math.cos(el_b) * math.cos(az_a - az_b)
    return math.acos(max(-1.0, min(1.0, cos_d)))


def geometric_profile_distance(
    achieved: Mapping[str, float],
    goal: Mapping[str, float],
    intr: CameraIntrinsics,
) -> float:
    """Weight-free product metric on the camera-subject geometry (radians)."""
    a = profile_to_geometry(achieved, intr)
    g = profile_to_geometry(goal, intr)
    d_view = _great_circle(a["az"], a["el"], g["az"], g["el"])
    d_size = a["size"] - g["size"]
    d_aim = math.hypot(a["aim_x"] - g["aim_x"], a["aim_y"] - g["aim_y"])
    return math.sqrt(d_view * d_view + d_size * d_size + d_aim * d_aim)


def profile_distance_value(
    achieved: Mapping[str, float],
    goal: Mapping[str, float],
    intr: CameraIntrinsics,
) -> float:
    """Value target: negative geometric profile distance (0 at the goal, <0 away)."""
    return -geometric_profile_distance(achieved, goal, intr)


# --- Legacy weighted-L2 score distance (eval pose-proxy / ablations) ----------


def score_distance(
    achieved: np.ndarray,
    goal: np.ndarray,
    keys: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Weighted L2 between achieved and goal profile vectors.

    By default both vectors are normalized to [-1, 1] per key before the L2,
    which keeps keys with different physical scales (degrees vs. percent)
    comparable. Pass `normalize=False` if the inputs are already on a shared
    scale.
    """
    keys = goal_keys(keys)
    a = np.asarray(achieved, dtype=np.float32)
    g = np.asarray(goal, dtype=np.float32)
    if normalize:
        a = normalize_goal(a, keys)
        g = normalize_goal(g, keys)
    diff = a - g
    if weights is None:
        return float(np.sqrt(np.mean(diff * diff)))
    w = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=np.float32)
    return float(np.sqrt(np.sum(w * diff * diff) / np.sum(w)))


def score_distance_reward(
    achieved: np.ndarray,
    goal: np.ndarray,
    keys: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Negative of `score_distance` — higher = better. Suitable as a value target."""
    return -score_distance(achieved, goal, keys, weights, normalize=normalize)
