"""The one place where our conventions meet MAVSDK's.

This repo has had an inverted yaw three separate times (the GPT-5 agent prompt, UNIC
and `src/policy/llm_policy/`), so the boundary gets a single named function and a test
file rather than an inline minus sign at each call site.

Ours (`VelocityCommand`, and `src/policy/astra/pilot.py`):
    vx forward+, vy LEFT+, vz UP+, yaw_rate LEFT+ (counter-clockwise from above)

MAVSDK `VelocityBodyYawspeed(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s)`:
    forward+, RIGHT+, DOWN+, yawspeed CLOCKWISE+
    https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html

So three of the four axes invert. Only `vx` survives unchanged.
"""

from __future__ import annotations

from .commands import VelocityCommand

#: Each flips because MAVSDK measures the opposite direction as positive.
RIGHT_SIGN = -1.0      # ours is left-positive
DOWN_SIGN = -1.0       # ours is up-positive
YAWSPEED_SIGN = -1.0   # ours is counter-clockwise-positive


def to_mavsdk_body(cmd: VelocityCommand) -> tuple[float, float, float, float]:
    """Our command -> `(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s)`.

    Returns a plain tuple rather than a MAVSDK type so this module (and its tests)
    import without MAVSDK installed; `offboard` wraps it.
    """
    return (float(cmd.vx),
            RIGHT_SIGN * float(cmd.vy),
            DOWN_SIGN * float(cmd.vz),
            YAWSPEED_SIGN * float(cmd.yaw_rate))


def from_mavsdk_body(forward: float, right: float, down: float, yawspeed: float) -> VelocityCommand:
    """Inverse of `to_mavsdk_body`, for tests and for logging what was actually sent."""
    return VelocityCommand(vx=float(forward),
                           vy=RIGHT_SIGN * float(right),
                           vz=DOWN_SIGN * float(down),
                           yaw_rate=YAWSPEED_SIGN * float(yawspeed),
                           source="mavsdk")
