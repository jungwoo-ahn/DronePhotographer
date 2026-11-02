"""The chain of order: who gets to move the drone, and who gets overruled.

The policy is an ADVISOR, not a pilot. It proposes; the supervisor disposes. Every
command converges here, is checked for freshness, clamped to the envelope, and only
then handed to MAVSDK. Nothing else in this package is allowed to call `offboard`.

Priority, highest first:

  1. E-stop          -- operator pressed it. Hover now, land next. Latched: nothing
                        clears it but an explicit `clear_estop()`, so a late-arriving
                        policy command cannot un-stop the aircraft.
  2. Safety envelope -- clamps speed, altitude, geofence (see `safety`).
  3. Staleness       -- the command's TTL expired, i.e. we have lost the policy link.
                        Hover. This is the case MAVSDK silently hides, because its
                        offboard plugin keeps resending the last setpoint at 20 Hz
                        whether or not anyone is still producing commands.
  4. Policy command  -- the freshest one submitted.
  5. Nothing yet     -- hover.

The supervisor is deliberately synchronous and clock-injectable: it is pure decision
logic, so the tests can drive it through a whole link failure in microseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .commands import HOVER, VelocityCommand
from .safety import SafetyLimits, VehicleState, clamp


@dataclass(frozen=True)
class Decision:
    """What the supervisor decided, and why -- `reason` exists to be logged."""
    command: VelocityCommand
    reason: str
    estopped: bool = False
    stale: bool = False
    clamped: tuple[str, ...] = ()

    @property
    def should_land(self) -> bool:
        return self.estopped


class Supervisor:
    def __init__(self, limits: SafetyLimits | None = None, clock=time.monotonic):
        self.limits = limits or SafetyLimits()
        self._clock = clock
        self._latest: VelocityCommand | None = None
        self._estop = False
        self._accepted = 0
        self._rejected_stale = 0

    # -- inputs ---------------------------------------------------------------
    def submit(self, cmd: VelocityCommand | None) -> None:
        """Offer a command. `None` (a failed fetch) is ignored, letting the old one age out.

        Ignoring rather than hovering is deliberate: one dropped HTTP request inside
        the TTL should not jerk the drone to a stop, but repeated failures will,
        because the last good command simply expires.
        """
        if cmd is None:
            return
        self._latest = cmd.stamp(self._clock())
        self._accepted += 1

    def estop(self) -> None:
        self._estop = True

    def clear_estop(self) -> None:
        self._estop = False

    # -- the ladder -----------------------------------------------------------
    def decide(self, state: VehicleState) -> Decision:
        now = self._clock()

        if self._estop:
            return Decision(HOVER, "E-STOP latched", estopped=True)

        cmd = self._latest
        if cmd is None:
            return Decision(HOVER, "no command yet")

        if not cmd.is_fresh(now):
            self._rejected_stale += 1
            return Decision(HOVER, f"command stale ({cmd.age_s(now):.2f}s > {cmd.ttl_s:.2f}s "
                                   f"TTL) -- policy link lost?", stale=True)

        safe, reasons = clamp(cmd, state, self.limits)
        if reasons:
            return Decision(safe, "clamped: " + "; ".join(reasons), clamped=tuple(reasons))
        return Decision(safe, f"policy cmd seq={cmd.seq}")

    @property
    def stats(self) -> dict:
        return {"accepted": self._accepted, "rejected_stale": self._rejected_stale,
                "estopped": self._estop}
