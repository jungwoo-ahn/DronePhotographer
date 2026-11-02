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

from .commands import VelocityCommand
from .conventions import to_mavsdk_body
from .supervisor import Decision

#: PX4 needs >2 Hz; 20 Hz matches what the MAVSDK plugin resends at anyway and keeps
#: the supervisor's reaction time (one tick) well under the TTL.
CONTROL_HZ = 20.0


def _mavsdk():
    try:
        import mavsdk  # noqa: F401
        from mavsdk import System
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        return System, VelocityBodyYawspeed, OffboardError
    except ImportError as e:  # pragma: no cover - depends on the laptop's env
        raise RuntimeError(
            "mavsdk is not installed. This module only runs on the laptop next to PX4:\n"
            "    pip install mavsdk\n"
            "The policy server does not need it."
        ) from e


class OffboardDriver:
    """Thin async wrapper: connect, arm, hold offboard, send velocities, land."""

    def __init__(self, system_address: str = "udp://:14540"):
        self.system_address = system_address
        self._drone = None
        self._VBY = None
        self._OffboardError = None
        self.started = False

    async def connect(self, timeout_s: float = 30.0):
        System, VBY, OffboardError = _mavsdk()
        self._VBY, self._OffboardError = VBY, OffboardError
        self._drone = System()
        await self._drone.connect(system_address=self.system_address)
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                break
        return self._drone

    async def wait_until_ready(self):
        """Block until PX4 reports a usable position estimate.

        Velocity setpoints need a valid velocity estimate, so arming before the EKF has
        converged produces a drone that accepts commands and then refuses to move.
        """
        async for health in self._drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                return

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
        try:
            await self.hover()
            if self.started:
                await self._drone.offboard.stop()
                self.started = False
        finally:
            await self._drone.action.land()

    async def state(self):
        """Current `VehicleState`, with altitude converted out of PX4's down-positive NED."""
        from .safety import VehicleState
        async for p in self._drone.telemetry.position_velocity_ned():
            return VehicleState(altitude=-p.position.down_m,   # NED down+ -> up+
                                north=p.position.north_m, east=p.position.east_m, ok=True)
        return VehicleState(ok=False)
