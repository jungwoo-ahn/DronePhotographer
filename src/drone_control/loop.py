"""The two-rate control loop.

Fast lane (`CONTROL_HZ`, 20 Hz): decide + send. Keeps PX4 in offboard and reacts to a
dead link within one tick.
Slow lane (`policy_hz`, ~1-2 Hz): ask the command source for a new opinion.

They are separate because the policy is ~100x slower than PX4's deadline. Running the
source inside the fast lane would either throttle the loop below PX4's 2 Hz floor
(dropping out of offboard mid-flight) or block it entirely on one slow HTTP call.
"""

from __future__ import annotations

import asyncio
import time

from .offboard import CONTROL_HZ
from .safety import VehicleState
from .supervisor import Supervisor


async def run(driver, supervisor: Supervisor, source, *, policy_hz: float = 2.0,
              duration_s: float | None = None, log=print, clock=time.monotonic):
    """Fly until `duration_s` elapses, the source is exhausted, or an e-stop lands.

    `driver` is an `OffboardDriver` (or anything with the same async surface, which is
    what the tests use). Returns the supervisor stats.
    """
    dt = 1.0 / CONTROL_HZ
    policy_dt = 1.0 / max(policy_hz, 1e-6)
    t0 = clock()
    next_poll = t0
    pending: asyncio.Task | None = None
    last_reason = None

    while True:
        now = clock()
        if duration_s is not None and now - t0 >= duration_s:
            log(f"[loop] duration {duration_s}s reached")
            break

        try:
            state = await driver.state()
        except Exception as e:                      # telemetry hiccup != crash
            log(f"[loop] telemetry failed ({e}); treating as unsafe")
            state = VehicleState(ok=False)

        # Slow lane: kick off a source query without blocking the fast lane.
        if now >= next_poll and pending is None:
            next_poll = now + policy_dt
            pending = asyncio.ensure_future(
                asyncio.to_thread(source.next_command, state, now))
        if pending is not None and pending.done():
            try:
                supervisor.submit(pending.result())
            except Exception as e:
                log(f"[loop] source raised ({e}); ignoring, command will age out")
            pending = None

        decision = supervisor.decide(state)
        await driver.send(decision)

        # Log transitions, not every tick -- 20 Hz of identical lines hides the event
        # you actually care about.
        if decision.reason != last_reason:
            log(f"[loop] {decision.reason}")
            last_reason = decision.reason

        if decision.should_land:
            log("[loop] e-stop -> landing")
            break

        await asyncio.sleep(dt)

    if pending is not None:
        pending.cancel()
    return supervisor.stats
