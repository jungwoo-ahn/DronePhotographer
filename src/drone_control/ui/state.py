"""The snapshot the UI renders. Written by the control loop, read by HTTP handlers.

Thread-safe by being boring: the loop builds a plain dict under a lock and the server
hands out the last one. No shared mutable objects cross the boundary, so a slow
browser can never stall the 20 Hz control loop -- the one property that matters here.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from ..conventions import to_mavsdk_body


class UiState:
    def __init__(self, limits=None, max_events: int = 120):
        self._lock = threading.Lock()
        self._snap: dict = {"status": "starting"}
        self._events: deque = deque(maxlen=max_events)
        self._limits = limits
        self._frame_jpeg: bytes | None = None
        self._last_reason: str | None = None
        self._cmd_times: deque = deque(maxlen=30)

    # -- written by the control loop -----------------------------------------
    def publish(self, *, decision, state, camera=None, client=None, source_name="",
                now: float | None = None, started_at: float | None = None):
        now = time.monotonic() if now is None else now
        cmd = decision.command
        f, r, d, y = to_mavsdk_body(cmd)
        self._cmd_times.append(now)

        frame = camera.latest() if camera is not None else None
        frame_age = frame.age_s(now) if frame is not None else None
        video_ok = bool(frame is not None and camera is not None
                        and frame.is_fresh(camera.max_age_s, now))

        rung = ("estop" if decision.estopped else
                "stale" if decision.stale else
                "envelope" if decision.clamped else
                "policy" if not cmd.is_hover() else "idle")

        snap = {
            "status": "flying",
            "t": now,
            "elapsed": None if started_at is None else now - started_at,
            "rung": rung,
            "reason": decision.reason,
            "source": source_name,
            "links": {
                "mavlink": {"ok": bool(state.ok)},
                "video": {"ok": video_ok,
                          "age_ms": None if frame_age is None else frame_age * 1000.0,
                          "present": camera is not None,
                          "reconnects": getattr(camera, "reconnects", 0) if camera else 0,
                          "frames": getattr(camera, "frames_read", 0) if camera else 0},
                "policy": {"ok": not decision.stale,
                           "present": client is not None,
                           "failures": getattr(client, "failures", 0) if client else 0,
                           "calls": getattr(client, "calls", 0) if client else 0},
            },
            # Both frames, side by side: this panel is where a sign error becomes
            # obvious instead of becoming a crash.
            "command": {"vx": cmd.vx, "vy": cmd.vy, "vz": cmd.vz, "yaw_rate": cmd.yaw_rate,
                        "seq": cmd.seq, "note": cmd.note, "ttl_s": cmd.ttl_s},
            "sent": {"forward": f, "right": r, "down": d, "yawspeed": y},
            "clamped": list(decision.clamped),
            "telemetry": {"altitude": state.altitude, "north": state.north,
                          "east": state.east,
                          "radius": (state.north ** 2 + state.east ** 2) ** 0.5},
            "limits": self._limits_dict(),
            "hz": self._rate(now),
        }
        if decision.reason != self._last_reason:
            self._events.appendleft({"t": now, "text": decision.reason, "rung": rung})
            self._last_reason = decision.reason
        snap["events"] = list(self._events)[:40]

        with self._lock:
            self._snap = snap

    def publish_frame(self, jpeg: bytes | None):
        with self._lock:
            self._frame_jpeg = jpeg

    def _rate(self, now):
        if len(self._cmd_times) < 2:
            return None
        span = self._cmd_times[-1] - self._cmd_times[0]
        return (len(self._cmd_times) - 1) / span if span > 0 else None

    def _limits_dict(self):
        L = self._limits
        if L is None:
            return {}
        return {"max_horizontal_speed": L.max_horizontal_speed,
                "max_vertical_speed": L.max_vertical_speed,
                "max_yaw_rate": L.max_yaw_rate,
                "min_altitude": L.min_altitude, "max_altitude": L.max_altitude,
                "max_radius": L.max_radius}

    # -- read by the HTTP handlers -------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg
