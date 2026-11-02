"""The command schema shared by the policy server and the laptop.

One rule drives the design: a command is only valid for a short while. The policy runs
on a remote GPU over a network link that can stall, and MAVSDK keeps resending the last
setpoint at 20 Hz regardless (see `offboard`), so a dropped link does NOT stop the
drone -- it flies the stale command forever. Every command therefore carries a TTL and
the supervisor refuses to use an expired one.

Freshness is measured from when the LAPTOP received the command, never from the
server's own timestamp: the two machines do not share a clock, and a few seconds of
skew would either expire every command instantly or never expire any. `received_at`
is stamped locally on arrival, with `time.monotonic` so a wall-clock adjustment
(NTP step, DST) cannot make a stale command look fresh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

# How long a policy command stays valid. The policy answers at roughly 0.5-2 Hz, so
# 1.0 s tolerates one missed round trip and stops the drone on the second.
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

    def to_json(self) -> dict:
        return {"vx": self.vx, "vy": self.vy, "vz": self.vz, "yaw_rate": self.yaw_rate,
                "ttl_s": self.ttl_s, "source": self.source, "seq": self.seq, "note": self.note}

    @classmethod
    def from_json(cls, d: dict, source: str | None = None) -> "VelocityCommand":
        """Build from a server reply. Unknown keys are ignored; missing ones hover.

        Anything the server omits defaults to zero, so a malformed or truncated reply
        degrades to hover rather than to an arbitrary velocity.
        """
        def num(k):
            try:
                return float(d.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return cls(vx=num("vx"), vy=num("vy"), vz=num("vz"), yaw_rate=num("yaw_rate"),
                   ttl_s=float(d.get("ttl_s") or DEFAULT_TTL_S),
                   source=source or str(d.get("source") or "policy"),
                   seq=int(d.get("seq") or 0), note=str(d.get("note") or ""))


#: The command every failure path falls back to.
HOVER = VelocityCommand(source="hover", ttl_s=float("inf"), note="default hover")
