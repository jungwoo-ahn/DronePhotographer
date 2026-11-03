"""The policy's pose action -> what a drone with a pitch-only gimbal can actually do.

The policy predicts `[Δtranslation_cam(3), rot6d(6), shoot]`, the same 10D action the
rollouts apply with `apply_action_9d`. Rollouts remove roll from the result, which leaves
a camera with two rotational axes, and on the aircraft they belong to different machines:

    azimuth   -> the airframe's yaw
    elevation -> the gimbal's tilt

**The current tilt is an input, not a detail.** The translation is in the *tilted*
camera's frame, so "forward" at -30° means forward and down. Reading it as `vx` would fly
the aircraft level past a subject the camera is looking down at.

Runs beside the policy on the server, which is why it lives here and not in
`src/drone_control`: that package has to import without numpy or `src.policy`. Speeds are
in drone_control's conventions, and its `conventions.to_mavsdk_body` remains the only
place that converts them for MAVSDK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.policy.common.action_repr import POSE_DIM, apply_action_9d

#: The SIYI A2 mini's mechanical range, from its manual.
TILT_MIN_DEG = -90.0
TILT_MAX_DEG = 25.0


@dataclass(frozen=True)
class GimbalMove:
    """One action as rates held for `dt`, and the gimbal angle it should end on."""
    vx: float                   # m/s forward, level
    vy: float                   # m/s left
    vz: float                   # m/s up
    yaw_rate: float             # deg/s left+ (counter-clockwise)
    tilt_rate: float            # deg/s up+, after the limits below
    tilt_target: float          # deg, inside the gimbal's range
    tilt_limited: bool          # the policy asked for an angle the gimbal cannot reach
    shoot: bool


def action_to_move(action, tilt_deg: float, dt: float) -> GimbalMove:
    """Fly `action` over `dt` seconds, from a camera currently tilted `tilt_deg` from level.

    Works in a frame with the aircraft at the origin heading along +X, so nothing here
    depends on where it really is. The rotation goes through `apply_action_9d`, which is
    what makes this the same motion the rollouts were scored on.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    t = math.radians(tilt_deg)
    forward = np.array([math.cos(t), 0.0, math.sin(t)], dtype=np.float32)
    up = np.array([-math.sin(t), 0.0, math.cos(t)], dtype=np.float32)
    p, f, _ = apply_action_9d(np.zeros(3, dtype=np.float32), forward, up, a[:POSE_DIM],
                              upright=True)

    # +X forward, +Y left, +Z up is already our command frame. Azimuth is measured the
    # same way, so a positive yaw turns left with no sign flip.
    yaw = math.degrees(math.atan2(float(f[1]), float(f[0])))
    wanted = math.degrees(math.asin(max(-1.0, min(1.0, float(f[2])))))
    target = max(TILT_MIN_DEG, min(TILT_MAX_DEG, wanted))
    return GimbalMove(vx=float(p[0]) / dt, vy=float(p[1]) / dt, vz=float(p[2]) / dt,
                      yaw_rate=yaw / dt, tilt_rate=(target - tilt_deg) / dt,
                      tilt_target=target, tilt_limited=target != wanted,
                      shoot=a.shape[0] > POSE_DIM and float(a[POSE_DIM]) > 0.5)
