"""Human in the loop: the policy proposes a move, the operator runs it.

`ApprovalGate` wraps any `CommandSource` and inverts its relationship with the drone.
The inner source still proposes whenever it likes, but nothing reaches the supervisor
until someone clicks. Until then the aircraft holds.

Three properties matter more than the feature itself:

* **A click starts a move, the clock ends it.** `accept()` makes the proposal live for
  `hold_s` and no longer. The gate keeps handing the supervisor that one command while
  it is live, so the supervisor's own TTL still ends the move if the laptop dies
  mid-flight. There is no path that leaves a velocity running because a browser tab
  froze.
* **The panel can refuse a move; it cannot write one.** `accept()` releases exactly
  what the inner source proposed, and the gate exposes no way to supply a command.
  A UI bug can decline to fly or it can fly the policy's own suggestion early. It
  cannot invent a velocity.
* **A proposal goes stale too.** The view the policy answered about is seconds old by
  the time a human reads it, so a proposal nobody acts on expires rather than waiting.

The gate is itself a `CommandSource`, so `loop.run` needs no changes: it is just a
source that usually says nothing.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .commands import VelocityCommand
from .conventions import to_mavsdk_body
from .safety import SafetyLimits, clamp

DEFAULT_HOLD_S = 1.5        # how long one accepted move flies before the aircraft holds
DEFAULT_OFFER_S = 6.0       # how long a proposal stays on offer before it is withdrawn


MIN_SPEED = 0.01        # m/s: 1.5 cm over a 1.5 s move. The real policy steps a few cm at a time
MIN_YAW_RATE = 0.5      # deg/s
SHARE = 0.3             # a smaller part of the move than this, next to the rest, goes unnamed


def describe_move(cmd: VelocityCommand) -> str:
    """One plain sentence for a velocity, so the operator reads intent, not axes.

    "Hold still" only when nothing moves: the real policy's steps are a few centimetres, and
    calling eight of them "hold still" once hid 30 cm of travel.
    """
    fwd, left, up, yaw = cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate
    peak = max(abs(fwd), abs(left), abs(up))
    named = lambda v: abs(v) >= MIN_SPEED and abs(v) >= SHARE * peak    # noqa: E731
    turning = abs(yaw) >= MIN_YAW_RATE
    sliding = named(left)
    parts: list[str] = []

    # A slide and a turn the same way round is an orbit, which is what the shot is
    # usually made of; saying "slide left and turn left" hides that.
    if sliding and turning and (left > 0) == (yaw > 0):
        parts.append("Arc " + ("left" if left > 0 else "right") + " around the subject")
    else:
        if sliding:
            parts.append("Slide " + ("left" if left > 0 else "right"))
        if turning:
            parts.append(("Turn " if not parts else "turning ") + ("left" if yaw > 0 else "right"))
    if named(fwd):
        word = "closer" if fwd > 0 else "back"
        parts.append(("Move " if not parts else "moving ") + word)
    if named(up):
        parts.append(("Climb" if up > 0 else "Descend") if not parts
                     else ("climbing" if up > 0 else "descending"))
    if not parts:
        return "Hold still"
    head, rest = parts[0], parts[1:]
    if not rest:
        return head
    if len(rest) == 1:
        return f"{head}, {rest[0]}"
    return f"{head}, " + ", ".join(rest[:-1]) + f" and {rest[-1]}"



class ApprovalGate:
    """A command source that only speaks once a human has agreed with it."""

    name = "approval"

    def __init__(self, source, *, limits: SafetyLimits | None = None,
                 hold_s: float = DEFAULT_HOLD_S, offer_s: float = DEFAULT_OFFER_S,
                 clock=time.monotonic, max_record: int = 40):
        self.source = source
        self.limits = limits or SafetyLimits()
        self.hold_s = float(hold_s)
        self.offer_s = float(offer_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._live: dict | None = None
        self._state = None                  # last vehicle state, for the clamp preview
        self._record: deque = deque(maxlen=max_record)
        self.accepted = self.skipped = self.expired = 0

    # -- the control loop's side ---------------------------------------------
    def hold(self) -> VelocityCommand:
        """What the aircraft is told between moves: hover, and say why.

        An explicit hover rather than `None`, because holding here is a decision, not
        an absence. The supervisor's TTL is still the backstop: if this gate stops
        being asked at all, the last hover ages out and the supervisor hovers anyway.
        """
        return VelocityCommand(source=self.name, note="holding for approval")

    def next_command(self, state, now: float) -> VelocityCommand | None:
        """Called from the slow lane. Returns the live move, or a hold."""
        ask = False
        with self._lock:
            self._state = state
            if self._live is not None:
                if now < self._live["until"]:
                    return self._live["cmd"]
                self._close(self._live, "flown", now)
                self._live = None
            if self._pending is not None and now - self._pending["at"] > self.offer_s:
                self._close(self._pending, "expired", now)
                self._pending = None
                self.expired += 1
            ask = self._pending is None
        if not ask:
            return self.hold()

        # Outside the lock: this blocks on the camera and on the policy server, and the
        # UI thread must never wait behind it.
        cmd = self.source.next_command(state, self._clock() if now is None else now)
        current = getattr(self.source, "still_current", None)
        if cmd is not None:
            with self._lock:
                # Checked under this lock so that a shot picked meanwhile, which takes the
                # offer back under it too, cannot leave a move for the old shot on offer.
                if current is not None and not current(cmd):
                    return self.hold()
                if self._pending is None and self._live is None:
                    self._pending = {"cmd": cmd, "at": now, "wall": time.time()}
        return self.hold()

    # -- the panel's side -----------------------------------------------------
    def accept(self, now: float | None = None) -> bool:
        """Run the move that is on offer. Returns False if there was nothing to run."""
        now = self._clock() if now is None else now
        ready = getattr(self.source, "ready_to_run", None)
        if ready is not None and not ready():
            return False
        with self._lock:
            if self._pending is None:
                return False
            self._live = {"cmd": self._pending["cmd"], "at": self._pending["at"],
                          "wall": self._pending["wall"], "until": now + self.hold_s,
                          "ran_at": now}
            cmd = self._live["cmd"]
            self._pending = None
            self.accepted += 1
        started = getattr(self.source, "move_started", None)
        if started is not None:
            started(cmd)
        return True

    def skip(self) -> bool:
        """Refuse the move on offer, and what depended on it, so the next comes from a newer view."""
        now = self._clock()
        with self._lock:
            if self._pending is None:
                return False
            self._close(self._pending, "skipped", now)
            self._pending = None
            self.skipped += 1
        drop = getattr(self.source, "drop_sequence", None)
        if drop is not None:
            drop()
        return True

    def withdraw(self, outcome: str = "withdrawn") -> bool:
        """Take back the move on offer, because what it was for has changed (a new shot).

        Not a skip: the operator did not refuse the move, it stopped applying.
        """
        now = self._clock()
        with self._lock:
            if self._pending is None:
                return False
            self._close(self._pending, outcome, now)
            self._pending = None
        return True

    # -- what the panel draws --------------------------------------------------
    def snapshot(self, now: float | None = None) -> dict:
        now = self._clock() if now is None else now
        with self._lock:
            pending, live, state = self._pending, self._live, self._state
            record = list(self._record)
        offer = None
        if pending is not None:
            offer = self._describe(pending, state)
            offer["expires_in_s"] = max(0.0, self.offer_s - (now - pending["at"]))
        running = None
        if live is not None:
            running = self._describe(live, state)
            running["ends_in_s"] = max(0.0, live["until"] - now)
        return {"offer": offer, "running": running, "record": record,
                "hold_s": self.hold_s, "offer_s": self.offer_s,
                "accepted": self.accepted, "skipped": self.skipped,
                "expired": self.expired}

    def _describe(self, entry: dict, state) -> dict:
        """A move as the panel needs it: words, our numbers, MAVSDK's, and the clamp."""
        cmd = entry["cmd"]
        flown, notes = (clamp(cmd, state, self.limits) if state is not None
                        else (cmd, []))
        f, r, d, y = to_mavsdk_body(flown)
        return {
            "text": describe_move(cmd),
            "wall": entry["wall"],
            "seq": cmd.seq,
            "note": cmd.note,
            "command": {"vx": flown.vx, "vy": flown.vy, "vz": flown.vz,
                        "yaw_rate": flown.yaw_rate},
            "asked": {"vx": cmd.vx, "vy": cmd.vy, "vz": cmd.vz, "yaw_rate": cmd.yaw_rate},
            "sent": {"forward": f, "right": r, "down": d, "yawspeed": y},
            "clamped": list(notes),
            "hold_s": self.hold_s,
        }

    def _close(self, entry: dict, outcome: str, now: float):
        self._record.appendleft({"wall": time.time(), "outcome": outcome,
                                 "text": describe_move(entry["cmd"]),
                                 "seq": entry["cmd"].seq})


__all__ = ["ApprovalGate", "describe_move", "DEFAULT_HOLD_S", "DEFAULT_OFFER_S"]
