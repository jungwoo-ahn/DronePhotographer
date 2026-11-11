"""Where commands come from. Swap freely -- the supervisor cannot tell them apart.

A source only ever *proposes*. It has no access to the driver, and returning `None`
(no opinion, or a failed fetch) is always legal: the supervisor keeps the previous
command until its TTL runs out, then hovers.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Protocol

from .actions import action_to_move
from .approval import describe_move
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


def plan_path(actions, tilt_deg: float, move_s: float, *, follow_tilt: bool = True) -> list[dict]:
    """Where a sequence would take the aircraft, step by step, if every action is run.

    In the aircraft's frame when the sequence was asked for: x forward, y left, z up, in
    metres. Each action is converted from the tilt the one before it leaves the camera at,
    as `ActionSequenceSource` does when it hands them over, unless the camera does not move.
    """
    x = y = z = heading = 0.0
    tilt = float(tilt_deg)
    steps = []
    for n, action in enumerate(actions, 1):
        m = action_to_move(action, tilt, move_s)
        h = heading + math.radians(m.yaw_rate * move_s) / 2.0   # it turns while it moves
        dx, dy = m.vx * move_s, m.vy * move_s
        x += dx * math.cos(h) - dy * math.sin(h)
        y += dx * math.sin(h) + dy * math.cos(h)
        z += m.vz * move_s
        heading += math.radians(m.yaw_rate * move_s)
        if follow_tilt:
            tilt = m.tilt_target
        steps.append({"n": n, "x": x, "y": y, "z": z, "heading_deg": math.degrees(heading),
                      "tilt": tilt if follow_tilt else m.tilt_target, "shoot": m.shoot,
                      "text": describe_move(VelocityCommand(vx=m.vx, vy=m.vy, vz=m.vz,
                                                            yaw_rate=m.yaw_rate))})
    return steps


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
                 tilt_deg: float = 0.0, jpeg_quality: int = 80, view=None, hold=None):
        self.camera = camera
        self.view = view
        self.hold = hold            # aims the camera; without it the tilt stays at tilt_deg
        self._offered_tilt: float | None = None
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
        self._goal_n = 0            # which goal a forward pass was asked with
        self._asked_at: float | None = None
        self._last_pass_s: float | None = None
        self._answered_wall: float | None = None
        self._plan: list[dict] = []
        self._offered = 0           # how far through the plan the offers have got
        self._dropped = False
        self._last_offer: tuple = (None, -1)

    def _ask(self, timeout_s: float | None = None, goal: dict | None = None) -> list | None:
        frame = self.camera.fresh_frame()
        if frame is None:
            return None
        image = self.view(frame.image) if self.view is not None else frame.image
        jpeg = encode_jpeg(image, self.jpeg_quality)
        if jpeg is None:
            return None
        self.forward_passes += 1
        return self.client.request_actions(jpeg, goal or self.goal, timeout_s=timeout_s)

    def check(self, timeout_s: float) -> bool:
        """One forward pass on the ground, discarded: proves the whole path and warms up."""
        return self._ask(timeout_s) is not None

    def set_goal(self, goal: dict) -> None:
        """Ask for a different shot. What is left of the old sequence was for the old one."""
        with self._lock:
            self.goal = goal
            self._goal_n += 1
            self._actions.clear()
            self._plan, self._offered, self._dropped = [], 0, False
            self._asking = False        # an answer still on its way is for the old goal

    def drop_sequence(self) -> None:
        with self._lock:
            self._actions.clear()
            self._dropped = bool(self._plan)

    def status(self) -> dict:
        """What the model is doing, for the panel: thinking, or what it last answered."""
        now = time.monotonic()
        with self._lock:
            return {"asking": self._asking,
                    "asking_for_s": now - self._asked_at if self._asking else None,
                    "passes": self.forward_passes, "last_pass_s": self._last_pass_s,
                    "answered_wall": self._answered_wall,
                    "error": getattr(self.client, "last_error", None),
                    "plan": list(self._plan), "offered": self._offered,
                    "dropped": self._dropped, "move_s": self.move_s}

    def _fetch(self, goal: dict, goal_n: int) -> None:
        started = time.monotonic()
        actions = self._ask(goal=goal)
        with self._lock:
            if goal_n != self._goal_n:
                return                  # the operator picked another shot meanwhile
            self._asking = False
            if not actions:
                return
            self._last_pass_s = time.monotonic() - started
            self._answered_wall = time.time()
            self._actions.extend(actions)
            self._total = len(actions)
            tilt = self.hold.target if self.hold is not None else self.tilt_deg
            self._plan = plan_path(actions, tilt, self.move_s, follow_tilt=self.hold is not None)
            self._offered, self._dropped = 0, False

    def ready_to_run(self) -> bool:
        """An action on offer was worked out for a camera angle, so it needs that angle to run."""
        return self.hold is None or self.hold.settled()

    def move_started(self, cmd: VelocityCommand) -> None:
        """The operator ran what was offered, so the camera goes where that action says."""
        if self.hold is not None and self._offered_tilt is not None:
            self.hold.set_target(self._offered_tilt)

    def next_command(self, state, now: float) -> VelocityCommand | None:
        # Nothing is asked for or offered until the camera is where the next step assumes.
        if self.hold is not None and not self.hold.settled():
            return None
        tilt = self.hold.target if self.hold is not None else self.tilt_deg
        with self._lock:
            if not self._actions:
                if not self._asking:
                    self._asking, self._asked_at = True, time.monotonic()
                    threading.Thread(target=self._fetch, args=(self.goal, self._goal_n),
                                     name="forward-pass", daemon=True).start()
                return None
            step = self._total - len(self._actions) + 1
            action = self._actions.popleft()
            self._offered = step
            goal_n = self._goal_n
        m = action_to_move(action, tilt, self.move_s)
        self._offered_tilt = m.tilt_target
        verb = "The camera goes" if self.hold is not None else "The policy would tilt"
        note = f"Action {step} of {self._total}. {verb} to {round(m.tilt_target)}°."
        if m.shoot:
            note += " It says it has arrived."
        cmd = VelocityCommand(vx=m.vx, vy=m.vy, vz=m.vz, yaw_rate=m.yaw_rate,
                              source=self.name, seq=step, note=note)
        with self._lock:
            self._last_offer = (cmd, goal_n)
        return cmd

    def still_current(self, cmd: VelocityCommand) -> bool:
        """Whether `cmd` was worked out for the shot that is asked for now."""
        with self._lock:
            offer, goal_n = self._last_offer
            return offer is cmd and goal_n == self._goal_n
