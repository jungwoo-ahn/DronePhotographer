"""Coordinate transforms for VLM-guided placement adjustments."""

from __future__ import annotations

import math


def camera_to_world_adjustment(
    forward_m: float,
    right_m: float,
    up_m: float,
    camera_forward: list[float],
    camera_up: list[float],
) -> list[float]:
    """Convert camera-relative adjustment to world-space delta.

    Projects camera forward/right onto the ground plane (XY), keeps up as Z.
    """
    def normalize(v):
        length = math.sqrt(sum(x * x for x in v))
        return [x / length for x in v] if length > 1e-8 else v

    def cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    cam_fwd = list(camera_forward)
    cam_up_vec = list(camera_up)
    cam_right = cross(cam_fwd, cam_up_vec)
    ground_fwd = normalize([cam_fwd[0], cam_fwd[1], 0])
    ground_right = normalize([cam_right[0], cam_right[1], 0])

    return [
        forward_m * ground_fwd[0] + right_m * ground_right[0],
        forward_m * ground_fwd[1] + right_m * ground_right[1],
        up_m,
    ]
