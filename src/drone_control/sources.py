"""Where commands come from. Swap freely -- the supervisor cannot tell them apart.

A source only ever *proposes*. It has no access to the driver, and returning `None`
(no opinion, or a failed fetch) is always legal: the supervisor keeps the previous
command until its TTL runs out, then hovers.
"""

from __future__ import annotations

import math
import time
from typing import Protocol

from .camera import CameraStream, encode_jpeg
from .commands import VelocityCommand


class CommandSource(Protocol):
    name: str

    def next_command(self, state, now: float) -> VelocityCommand | None: ...


class HoverSource:
    """Does nothing, forever. Useful as the baseline when bringing up a new airframe."""
    name = "hover"

    def next_command(self, state, now: float) -> VelocityCommand | None:
        return VelocityCommand(source=self.name, note="idle")


class SquareSource:
    """Fly a square at constant speed: the milestone-1 stub that needs no policy.

    Exercises every axis the policy will eventually drive (two translation axes and
    yaw) with a trajectory whose correctness is obvious from the log, so a sign error
    shows up as a visibly wrong shape rather than as a plausible-looking drift.
    """
    name = "square"

    def __init__(self, speed: float = 1.0, leg_s: float = 4.0, turn_deg_s: float = 30.0,
                 turn_s: float = 3.0):
        self.speed, self.leg_s = speed, leg_s
        self.turn_deg_s, self.turn_s = turn_deg_s, turn_s
        self._t0: float | None = None

    def next_command(self, state, now: float) -> VelocityCommand | None:
        if self._t0 is None:
            self._t0 = now
        phase = (now - self._t0) % (4 * (self.leg_s + self.turn_s))
        leg = phase % (self.leg_s + self.turn_s)
        n = int(phase // (self.leg_s + self.turn_s))
        if leg < self.leg_s:
            return VelocityCommand(vx=self.speed, source=self.name, seq=n,
                                   note=f"leg {n + 1}/4")
        # Left turn: yaw_rate is LEFT-positive in our convention, so a square flown
        # with positive yaw goes counter-clockwise.
        return VelocityCommand(yaw_rate=self.turn_deg_s, source=self.name, seq=n,
                               note=f"turn {n + 1}/4")


class ScriptedSource:
    """Replay a fixed list of `(duration_s, VelocityCommand)` once, then hover."""
    name = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)
        self._t0: float | None = None

    def next_command(self, state, now: float) -> VelocityCommand | None:
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        for dur, cmd in self.steps:
            if t < dur:
                return cmd
            t -= dur
        return None     # exhausted -> let it expire into hover


class VisionPolicySource:
    """Camera frame + goal -> the policy server -> a velocity command.

    This is where the two remote links meet, and it enforces the rule that neither may
    be trusted on its own:

    * **No fresh frame -> no command.** Returning `None` lets the previous command age
      out through the supervisor's TTL and the drone hovers. The alternative -- sending
      the policy a stale image -- gets back a confident command computed from a view the
      drone has already flown past, which is far more dangerous than stopping.
    * **No reply -> no command.** `PolicyClient` already returns `None` on any failure.

    So one TTL mechanism covers both failure domains, rather than two half-watchdogs.
    """
    name = "vision-policy"

    def __init__(self, camera: CameraStream, client, goal: dict | None = None,
                 jpeg_quality: int = 80):
        self.camera = camera
        self.client = client
        self.goal = goal or {}
        self.jpeg_quality = jpeg_quality
        self.skipped_stale_frame = 0

    def next_command(self, state, now: float) -> VelocityCommand | None:
        frame = self.camera.fresh_frame(now)
        if frame is None:
            self.skipped_stale_frame += 1
            return None
        jpeg = encode_jpeg(frame.image, self.jpeg_quality)
        if jpeg is None:
            self.skipped_stale_frame += 1
            return None
        cmd = self.client.request_command(jpeg, state, self.goal)
        if cmd is None:
            return None
        return VelocityCommand(vx=cmd.vx, vy=cmd.vy, vz=cmd.vz, yaw_rate=cmd.yaw_rate,
                               ttl_s=cmd.ttl_s, source=self.name, seq=frame.seq,
                               note=cmd.note)
