"""The flight's own log: one JSON object per line, every one stamped with the laptop's clock.

What was asked for, what the model planned, each move offered and how it ended, each photo,
and twice a second where the aircraft and the camera were. With the recorder's video-time
reports in the same file, any moment of the video can be matched to the move that was flying.

    {"wall": 1794396790.65, "kind": "move", "outcome": "flown", "text": "Slide left", "seq": 1}

Built from the panel's snapshot, so it records exactly what the operator saw.
"""

from __future__ import annotations

import json
import threading
import time

STATE_EVERY_S = 0.5


class FlightLog:
    def __init__(self, path, clock=time.time):
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8", buffering=1)
        self._seen: set = set()
        self._state_at = None
        self._answered = None
        self._shot = None
        self._running = None
        self._rung = None

    def event(self, kind: str, **data):
        line = json.dumps({"wall": self._clock(), "kind": kind, **data}, default=str)
        with self._lock:
            if not self._f.closed:
                self._f.write(line + "\n")

    def tick(self, snap: dict):
        """Called with each panel snapshot. Writes what changed, and the state now and then."""
        if snap.get("status") != "flying":
            return
        move = snap.get("move") or {}
        for entry in reversed(move.get("record") or []):
            key = (entry.get("wall"), entry.get("seq"), entry.get("outcome"))
            if key not in self._seen:
                self._seen.add(key)
                self.event("move", outcome=entry.get("outcome"), text=entry.get("text"),
                           seq=entry.get("seq"))
        running = move.get("running")
        if running is not None and running.get("wall") != self._running:
            self._running = running.get("wall")
            self.event("run", text=running.get("text"), note=running.get("note"),
                       command=running.get("command"), seq=running.get("seq"))
        model = snap.get("model") or {}
        if model.get("answered_wall") and model["answered_wall"] != self._answered:
            self._answered = model["answered_wall"]
            self.event("plan", took_s=model.get("last_pass_s"), plan=model.get("plan"))
        if snap.get("shot") != self._shot:
            self._shot = snap.get("shot")
            self.event("shot", id=self._shot)
        if snap.get("rung") != self._rung:
            if snap.get("rung") == "estop":
                self.event("stop", reason=snap.get("reason"))
            self._rung = snap.get("rung")
        now = self._clock()
        if self._state_at is None or now - self._state_at >= STATE_EVERY_S:
            self._state_at = now
            self.event("state", telemetry=snap.get("telemetry"), command=snap.get("command"),
                       rung=snap.get("rung"), camera=snap.get("camera"),
                       video_age_ms=((snap.get("links") or {}).get("video") or {}).get("age_ms"))

    def close(self):
        with self._lock:
            self._f.close()


__all__ = ["FlightLog"]
