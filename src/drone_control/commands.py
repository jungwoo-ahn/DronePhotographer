"""The velocity command the laptop's control chain works in.

One rule drives the design: a command is only valid for a short while. MAVSDK keeps
resending the last setpoint at 20 Hz regardless (see `offboard`), so if this process stalls
the drone does NOT stop -- it flies the stale command forever. Every command therefore
carries a TTL and the supervisor refuses to use an expired one.

`received_at` is stamped when the supervisor takes the command, with `time.monotonic` so a
wall-clock adjustment (NTP step, DST) cannot make a stale command look fresh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

# How long a command stays valid. Sources are asked at roughly 2 Hz, so 1.0 s tolerates
# one missed turn and stops the drone on the second.
DEFAULT_TTL_S = 1.0


@dataclass(frozen=True)
class VelocityCommand:
    """A body-frame velocity request, in THIS repo's conventions, not MAVSDK's.

    `vy` is LEFT-positive, `vz` is UP-positive and `yaw_rate` is LEFT-positive
    (counter-clockwise seen from above) -- matching `src/policy/astra/pilot.py` and the
    5D action repr. MAVSDK wants right/down/clockwise; `conventions.to_mavsdk_body`
    is the single place that converts. Do not flip signs anywhere else.
    """

    vx: float = 0.0           # forward, m/s
    vy: float = 0.0           # LEFT, m/s
    vz: float = 0.0           # UP, m/s
    yaw_rate: float = 0.0     # LEFT / counter-clockwise, deg/s
    ttl_s: float = DEFAULT_TTL_S
    source: str = "unknown"
    seq: int = 0
    note: str = ""
    received_at: float = field(default_factory=time.monotonic)

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.received_at

    def is_fresh(self, now: float | None = None) -> bool:
        return self.age_s(now) < self.ttl_s

    def stamp(self, now: float | None = None) -> "VelocityCommand":
        """Re-stamp on arrival at the laptop. Call this the moment a command is received.

        `now` must come from the SAME clock that will later judge freshness -- the
        supervisor passes its injected clock. Reading `time.monotonic()` here instead
        would make the TTL disagree with any non-default clock, which silently
        disables the stale-link watchdog (and makes it untestable).
        """
        return replace(self, received_at=time.monotonic() if now is None else now)

    def is_hover(self) -> bool:
        return self.vx == self.vy == self.vz == self.yaw_rate == 0.0


#: The command every failure path falls back to.
HOVER = VelocityCommand(source="hover", ttl_s=float("inf"), note="default hover")
