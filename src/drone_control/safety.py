"""Envelope clamping: the last thing between a command and the airframe.

Everything here is advisory in the sense that it only ever makes a command *less*
energetic -- it clamps, zeroes or blocks, and never invents motion. That property is
what lets the supervisor apply it to any source (policy, stub, manual) without
reasoning about where the command came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .commands import VelocityCommand


@dataclass(frozen=True)
class SafetyLimits:
    """Conservative defaults sized for SITL, not for a real airframe.

    `max_horizontal_speed` is deliberately well under the Astra blog's 5 m/s: the
    policy was trained in a renderer with no dynamics, so its commands should be
    treated as directions, not as a velocity profile anyone validated.
    """
    max_horizontal_speed: float = 3.0   # m/s
    max_vertical_speed: float = 1.0     # m/s
    max_yaw_rate: float = 60.0          # deg/s
    min_altitude: float = 1.5           # m above takeoff, below this we refuse to descend
    max_altitude: float = 30.0          # m above takeoff
    max_radius: float = 50.0            # m from takeoff point


@dataclass(frozen=True)
class VehicleState:
    """What the supervisor needs to know to apply the envelope.

    `altitude` is metres ABOVE the takeoff point (up-positive), i.e. already converted
    out of PX4's down-positive NED by the telemetry layer.
    """
    altitude: float = 0.0
    north: float = 0.0
    east: float = 0.0
    ok: bool = True          # False when telemetry is missing/stale -> treat as unsafe


def clamp(cmd: VelocityCommand, state: VehicleState, limits: SafetyLimits
          ) -> tuple[VelocityCommand, list[str]]:
    """Return a command inside the envelope, plus the reasons anything was changed."""
    reasons: list[str] = []
    vx, vy, vz, yaw = cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate

    speed = math.hypot(vx, vy)
    if speed > limits.max_horizontal_speed:
        s = limits.max_horizontal_speed / speed      # scale, so direction is preserved
        vx, vy = vx * s, vy * s
        reasons.append(f"horizontal speed {speed:.2f}->{limits.max_horizontal_speed:.2f} m/s")

    if abs(vz) > limits.max_vertical_speed:
        vz = math.copysign(limits.max_vertical_speed, vz)
        reasons.append(f"vertical speed -> {vz:+.2f} m/s")

    if abs(yaw) > limits.max_yaw_rate:
        yaw = math.copysign(limits.max_yaw_rate, yaw)
        reasons.append(f"yaw rate -> {yaw:+.1f} deg/s")

    # Altitude floor/ceiling: block only the component that would make it worse, so the
    # drone can always fly back out of a violation it is already in.
    if state.altitude >= limits.max_altitude and vz > 0:
        vz = 0.0; reasons.append(f"at ceiling {limits.max_altitude} m, climb blocked")
    if state.altitude <= limits.min_altitude and vz < 0:
        vz = 0.0; reasons.append(f"at floor {limits.min_altitude} m, descent blocked")

    # Geofence: outside the radius, drop horizontal motion entirely. We cannot tell
    # "outward" from "inward" in the BODY frame without the vehicle's heading, and
    # guessing wrong would push it further out -- so refuse to translate at all and let
    # yaw still work, which is what lets an operator or the policy turn around.
    r = math.hypot(state.north, state.east)
    if r >= limits.max_radius and (vx or vy):
        vx = vy = 0.0
        reasons.append(f"outside geofence ({r:.0f} >= {limits.max_radius} m), translation blocked")

    if not state.ok:
        vx = vy = vz = 0.0
        reasons.append("telemetry not ok, holding position")

    if not reasons:
        return cmd, []
    return replace(cmd, vx=vx, vy=vy, vz=vz, yaw_rate=yaw), reasons
