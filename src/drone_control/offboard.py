"""MAVSDK offboard driver. The ONLY module that talks to the airframe.

Two PX4 facts shape this file:

1. PX4 leaves offboard mode unless setpoints keep arriving at **>2 Hz**
   (`COM_OF_LOSS_T`, then `COM_OBL_ACT`), and the stream must already be running
   *before* offboard is engaged -- hence `start()` sends a hover setpoint first.
   https://docs.px4.io/main/en/flight_modes/offboard

2. MAVSDK's offboard plugin **resends the last setpoint at 20 Hz by itself**.
   https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html
   That satisfies PX4's liveness check for free -- and is precisely why this package
   needs its own watchdog. PX4 cannot tell the difference between "the operator wants
   this velocity" and "the laptop died and MAVSDK is echoing a corpse". The supervisor's
   TTL is the only thing that can, so `send()` must be fed a `Decision`, never a raw
   policy command.

MAVSDK is imported lazily so the rest of the package (and all its tests) run on a
machine without it -- this server, for instance, which never flies anything.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace

from .commands import VelocityCommand
from .conventions import to_mavsdk_body
from .safety import VehicleState
from .supervisor import Decision

#: PX4 needs >2 Hz; 20 Hz matches what the MAVSDK plugin resends at anyway and keeps
#: the supervisor's reaction time (one tick) well under the TTL.
CONTROL_HZ = 20.0

#: Position older than this is treated as no position at all. A telemetry radio delivers
#: far less than SITL does, so this is sized for a slow link rather than a dead one.
TELEMETRY_MAX_AGE_S = 1.0

#: One landing step may not wait longer than this: the link may be what has failed.
LAND_STEP_TIMEOUT_S = 5.0


def _mavsdk():
    try:
        import mavsdk  # noqa: F401
        from mavsdk import System
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        return System, VelocityBodyYawspeed, OffboardError
    except ImportError as e:  # pragma: no cover - depends on the laptop's env
        raise RuntimeError(
            "mavsdk is not installed, or is v4, whose API is different. This module only "
            "runs on the laptop next to PX4:\n"
            '    pip install "mavsdk<4"\n'
            "The policy server does not need it."
        ) from e


class OffboardDriver:
    """Thin async wrapper: connect, arm, hold offboard, send velocities, land."""

    def __init__(self, system_address: str = "udpin://0.0.0.0:14540", clock=time.monotonic):
        self.system_address = system_address
        self._clock = clock
        self._position: VehicleState | None = None
        self._position_at = 0.0
        self._watcher: asyncio.Future | None = None
        self._drone = None
        self._VBY = None
        self._OffboardError = None
        self.started = False

    async def connect(self, timeout_s: float = 30.0):
        System, VBY, OffboardError = _mavsdk()
        self._VBY, self._OffboardError = VBY, OffboardError
        self._drone = System()

        async def link():
            await self._drone.connect(system_address=self.system_address)
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    return

        try:
            await asyncio.wait_for(link(), timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(f"no vehicle on {self.system_address} after {timeout_s:.0f} s; "
                               "check the radio, the baud rate and the URL") from None
        return self._drone

    async def wait_until_ready(self, timeout_s: float = 120.0):
        """Block until PX4 reports a usable position estimate.

        Velocity setpoints need a valid velocity estimate, so arming before the EKF has
        converged produces a drone that accepts commands and then refuses to move.
        """
        async def wait():
            async for health in self._drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    return

        try:
            await asyncio.wait_for(wait(), timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(f"no global position after {timeout_s:.0f} s. This needs a GPS "
                               "fix, so it will not pass indoors; see QGroundControl") from None

    async def arm_and_takeoff(self, altitude_m: float = 5.0):
        await self._drone.action.set_takeoff_altitude(float(altitude_m))
        await self._drone.action.arm()
        await self._drone.action.takeoff()

    async def start(self):
        """Engage offboard. Sends a hover setpoint first, as PX4 requires."""
        await self._drone.offboard.set_velocity_body(self._VBY(0.0, 0.0, 0.0, 0.0))
        try:
            await self._drone.offboard.start()
        except self._OffboardError as e:  # pragma: no cover - needs a live vehicle
            raise RuntimeError(f"offboard refused to start: {e._result.result}") from e
        self.started = True

    async def send(self, decision: Decision):
        """Send ONE supervisor decision. Takes a `Decision`, not a command, on purpose.

        Routing raw policy commands here would bypass the staleness and envelope checks,
        which is the whole safety story -- so the type makes that mistake awkward.
        """
        f, r, d, y = to_mavsdk_body(decision.command)
        await self._drone.offboard.set_velocity_body(self._VBY(f, r, d, y))

    async def hover(self):
        await self._drone.offboard.set_velocity_body(self._VBY(0.0, 0.0, 0.0, 0.0))

    async def stop_and_land(self):
        steps = [("hover", self.hover)]
        if self.started:
            steps.append(("leave offboard", self._drone.offboard.stop))
        steps.append(("land", self._drone.action.land))
        for name, step in steps:
            try:
                await asyncio.wait_for(step(), LAND_STEP_TIMEOUT_S)
            except Exception as e:
                # Carry on: failing to leave offboard must not cost us the land command.
                print(f"[offboard] {name} FAILED ({e!r}); take over from QGC or the RC")
        self.started = False
        if self._watcher is not None:
            self._watcher.cancel()

    async def _watch_position(self):
        async for p in self._drone.telemetry.position_velocity_ned():
            self._position = VehicleState(altitude=-p.position.down_m,   # NED down+ -> up+
                                          north=p.position.north_m, east=p.position.east_m,
                                          ok=True)
            self._position_at = self._clock()

    async def state(self):
        """Latest `VehicleState`, and `ok=False` once it is older than the limit.

        Never waits for the link. Waiting here once meant one silent radio froze the whole
        loop, with the e-stop and the TTL behind it, while MAVSDK echoed the last velocity.
        """
        if self._watcher is None:
            self._watcher = asyncio.ensure_future(self._watch_position())
        if self._position is None:
            return VehicleState(ok=False)
        if self._clock() - self._position_at > TELEMETRY_MAX_AGE_S:
            return replace(self._position, ok=False)
        return self._position


class BenchDriver(OffboardDriver):
    """Real telemetry and nothing else: for a bench test, with the propellers off.

    It can never arm, and every order stops here. This overrides the outputs rather than
    checking a flag inside them, so no path through the base class can reach the airframe.
    """

    async def arm_and_takeoff(self, altitude_m: float = 5.0):
        pass

    async def start(self):
        pass

    async def send(self, decision: Decision):
        pass

    async def hover(self):
        pass

    async def stop_and_land(self):
        if self._watcher is not None:
            self._watcher.cancel()


class DryRunDriver:
    """No aircraft and no MAVSDK: integrates the orders, so the panel can be tried anywhere."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._state = VehicleState(altitude=0.0)
        self._heading = 0.0                     # radians, counter-clockwise, like yaw_rate
        self._at = None

    async def connect(self, timeout_s: float = 30.0):
        pass

    async def wait_until_ready(self, timeout_s: float = 120.0):
        pass

    async def arm_and_takeoff(self, altitude_m: float = 5.0):
        self._state = replace(self._state, altitude=float(altitude_m))

    async def start(self):
        self._at = self._clock()

    async def send(self, decision: Decision):
        now = self._clock()
        dt, self._at = now - self._at, now
        c, h = decision.command, self._heading
        self._heading = h + math.radians(c.yaw_rate) * dt
        s = self._state
        self._state = replace(s, altitude=s.altitude + c.vz * dt,
                              north=s.north + (c.vx * math.cos(h) - c.vy * math.sin(h)) * dt,
                              east=s.east - (c.vx * math.sin(h) + c.vy * math.cos(h)) * dt)

    async def state(self):
        return self._state

    async def stop_and_land(self):
        self._state = replace(self._state, altitude=0.0)
