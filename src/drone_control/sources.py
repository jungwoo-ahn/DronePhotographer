"""Where commands come from. Swap freely -- the supervisor cannot tell them apart.

A source only ever *proposes*. It has no access to the driver, and returning `None`
(no opinion, or a failed fetch) is always legal: the supervisor keeps the previous
command until its TTL runs out, then hovers.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol

from .actions import action_to_move
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


class ActionSequenceSource:
    """Holds the server's sequence of actions and hands them over one at a time.

    Each call is one action. `ApprovalGate` asks for the next only once the last has flown,
    so this is what makes the operator approve every action in turn. The server is asked
    for a new forward pass only when the sequence has run out, and only with a fresh frame.
    That takes seconds, so it runs on its own thread: returning `None` meanwhile leaves the
    gate saying "hold", where blocking here would look to the supervisor like a lost link.

    The actions are relative to each other, so they only mean anything flown in order:
    `drop_sequence` throws away what is left rather than carrying on from a pose the
    aircraft never reached.
    """
    name = "action-sequence"

    def __init__(self, camera: CameraStream, client, goal: dict, *, move_s: float,
                 tilt_deg: float = 0.0, jpeg_quality: int = 80):
        self.camera = camera
        self.client = client
        self.goal = goal
        self.move_s = float(move_s)
        self.tilt_deg = float(tilt_deg)
        self.jpeg_quality = jpeg_quality
        self.forward_passes = 0
        self._lock = threading.Lock()
        self._actions: deque = deque()
        self._total = 0
        self._asking = False

    def _ask(self, timeout_s: float | None = None) -> list | None:
        frame = self.camera.fresh_frame()
        jpeg = encode_jpeg(frame.image, self.jpeg_quality) if frame is not None else None
        if jpeg is None:
            return None
        self.forward_passes += 1
        return self.client.request_actions(jpeg, self.goal, timeout_s=timeout_s)

    def check(self, timeout_s: float) -> bool:
        """One forward pass on the ground, discarded: proves the whole path and warms up."""
        return self._ask(timeout_s) is not None

    def drop_sequence(self) -> None:
        with self._lock:
            self._actions.clear()

    def _fetch(self) -> None:
        actions = self._ask()
        with self._lock:
            if actions:
                self._actions.extend(actions)
                self._total = len(actions)
            self._asking = False

    def next_command(self, state, now: float) -> VelocityCommand | None:
        with self._lock:
            if not self._actions:
                if not self._asking:
                    self._asking = True
                    threading.Thread(target=self._fetch, name="forward-pass", daemon=True).start()
                return None
            step = self._total - len(self._actions) + 1
            action = self._actions.popleft()
        m = action_to_move(action, self.tilt_deg, self.move_s)
        note = f"Action {step} of {self._total}. The policy would tilt to {m.tilt_target:.0f}°."
        if m.shoot:
            note += " It says it has arrived."
        return VelocityCommand(vx=m.vx, vy=m.vy, vz=m.vz, yaw_rate=m.yaw_rate,
                               source=self.name, seq=step, note=note)
