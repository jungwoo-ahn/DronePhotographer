"""One of the policy's actions -> what this drone and its pitch-only gimbal can do.

The server answers with raw 10D actions, `[Δtranslation_cam(3), rot6d(6), shoot]`, the
same ones the rollouts apply with `apply_action_9d`. Flying one is the laptop's business.
Roll is removed, which leaves two rotational axes that belong to different machines:

    azimuth   -> the airframe's yaw
    elevation -> the gimbal's tilt

**The current tilt is an input, not a detail.** The translation is in the *tilted* camera's
frame, so "forward" at -30° means forward and down. Reading it as `vx` would fly the
aircraft level past a subject the camera is looking down at.

Speeds are in our conventions (`commands`); `conventions.to_mavsdk_body` remains the only
place that converts them for MAVSDK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The SIYI A2 mini's mechanical range, from its manual.
TILT_MIN_DEG = -90.0
TILT_MAX_DEG = 25.0


@dataclass(frozen=True)
class Move:
    """One action as rates held for `dt`, and the gimbal angle it should end on."""
    vx: float                   # m/s forward, level
    vy: float                   # m/s left
    vz: float                   # m/s up
    yaw_rate: float             # deg/s left+ (counter-clockwise)
    tilt_rate: float            # deg/s up+, after the limits below
    tilt_target: float          # deg, inside the gimbal's range
    tilt_limited: bool          # the policy asked for an angle the gimbal cannot reach
    shoot: bool


def _forward_of_rot6d(r: list[float]) -> tuple[float, float, float]:
    """The third column of the rotation whose first two columns are `r`, orthonormalized."""
    a, b = r[:3], r[3:6]
    na = math.sqrt(sum(x * x for x in a))
    a = [x / na for x in a]
    d = sum(x * y for x, y in zip(a, b))
    b = [y - d * x for x, y in zip(a, b)]
    nb = math.sqrt(sum(x * x for x in b))
    b = [x / nb for x in b]
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def action_to_move(action, tilt_deg: float, dt: float) -> Move:
    """Fly `action` over `dt` seconds, from a camera currently tilted `tilt_deg` from level.

    Works in a frame with the aircraft at the origin heading along +X, +Y left, +Z up, which
    is already our command frame. The action's camera frame is OpenCV's: right, down, forward.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    t = math.radians(tilt_deg)
    s, c = math.sin(t), math.cos(t)
    right, down, forward = (0.0, -1.0, 0.0), (s, 0.0, -c), (c, 0.0, s)

    dx, dy, dz = action[:3]
    moved = [dx * right[i] + dy * down[i] + dz * forward[i] for i in range(3)]
    f = _forward_of_rot6d([float(x) for x in action[3:9]])
    aim = [f[0] * right[i] + f[1] * down[i] + f[2] * forward[i] for i in range(3)]

    # Azimuth is measured the same way round as yaw_rate, so left is positive with no flip.
    yaw = math.degrees(math.atan2(aim[1], aim[0]))
    wanted = math.degrees(math.asin(max(-1.0, min(1.0, aim[2]))))
    target = max(TILT_MIN_DEG, min(TILT_MAX_DEG, wanted))
    return Move(vx=moved[0] / dt, vy=moved[1] / dt, vz=moved[2] / dt,
                yaw_rate=yaw / dt, tilt_rate=(target - tilt_deg) / dt,
                tilt_target=target, tilt_limited=target != wanted,
                shoot=len(action) > 9 and float(action[9]) > 0.5)
