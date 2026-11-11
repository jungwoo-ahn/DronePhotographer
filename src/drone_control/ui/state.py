"""The snapshot the UI renders. Written by the control loop, read by HTTP handlers.

Thread-safe by being boring: the loop builds a plain dict under a lock and the server
hands out the last one. No shared mutable objects cross the boundary, so a slow
browser can never stall the 20 Hz control loop -- the one property that matters here.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from ..camera import encode_jpeg
from ..conventions import to_mavsdk_body


class UiState:
    def __init__(self, limits=None, max_events: int = 120, mode: str = ""):
        self._lock = threading.Lock()
        self._snap: dict = {"status": "starting"}
        self._events: deque = deque(maxlen=max_events)
        self._limits = limits
        self._mode = mode               # "BENCH" or "DRY RUN": the aircraft will not fly
        self._camera = None
        self._views: dict = {}
        self._frames: dict = {}         # which picture -> (frame seq, its JPEG)
        self._last_reason: str | None = None
        self._cmd_times: deque = deque(maxlen=30)
        self._sequence = None
        self._shots: list[dict] = []
        self._shot_id: str | None = None
        self._photos = None

    def attach_camera(self, camera, wide=None, model=None):
        """`wide` is what the operator looks at, `model` what the server is sent."""
        self._camera = camera
        self._views = {"wide": wide, "model": model}

    def attach_model(self, sequence, shots: list[dict], shot_id: str | None = None):
        """The policy's source, and the shots the operator can choose between."""
        self._sequence = sequence
        self._shots = list(shots)
        self._shot_id = shot_id if shot_id is not None else (shots[0]["id"] if shots else None)

    def shots(self) -> list[dict]:
        return list(self._shots)

    def attach_photos(self, taker):
        self._photos = taker

    def take_photo(self) -> dict | None:
        """Save the frame on screen, with the shot asked for and where everything was."""
        if self._photos is None:
            return None
        with self._lock:
            snap = dict(self._snap)
            shot_id = self._shot_id
        shot = next((s for s in self._shots if s["id"] == shot_id), None)
        hold = getattr(self._sequence, "hold", None)
        model = snap.get("model") or {}
        meta = {"mode": self._mode, "shot": shot_id,
                "profile": shot["profile"] if shot else None,
                "telemetry": snap.get("telemetry"),
                "camera_tilt_deg": hold.tilt if hold is not None else None,
                "camera_target_deg": hold.target if hold is not None else None,
                "model_step": model.get("offered"), "model_plan_len": len(model.get("plan") or [])}
        return self._photos.take(meta)

    def photo_path(self, name: str, which: str = "photo"):
        return None if self._photos is None else self._photos.path(name, which)

    def select_shot(self, shot_id: str) -> bool:
        """Point the policy at another shot. Only the goal changes: it still only proposes."""
        shot = next((s for s in self._shots if s["id"] == shot_id), None)
        if shot is None or self._sequence is None:
            return False
        self._sequence.set_goal({"profile": dict(shot["profile"])})
        with self._lock:
            self._shot_id = shot_id
        return True

    def preflight(self, phase: str):
        """What the pilot is doing on the ground, before the loop has anything to publish."""
        with self._lock:
            self._snap = {"status": "starting", "phase": phase, "mode": self._mode}

    # -- written by the control loop -----------------------------------------
    def publish(self, *, decision, state, camera=None, client=None, source_name="",
                gate=None, now: float | None = None, started_at: float | None = None):
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
            "mode": self._mode,
            "t": now,
            "wall": time.time(),
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
            # The move on offer, when the pilot is flying under approval. None means
            # the policy is driving directly and the panel has nothing to click.
            "move": None if gate is None else gate.snapshot(now),
            "model": None if self._sequence is None else self._sequence.status(),
            "shot": self._shot_id,
            "photos": None if self._photos is None else self._photos.summary(),
        }
        if decision.reason != self._last_reason:
            # `now` is monotonic, which is right for ages and wrong for a log: the panel
            # prints a UTC clock time beside every transition, so carry a wall clock too.
            self._events.appendleft({"t": now, "wall": time.time(),
                                     "text": decision.reason, "rung": rung})
            self._last_reason = decision.reason
        snap["events"] = list(self._events)[:40]

        with self._lock:
            self._snap = snap

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

    def frame(self, which: str = "wide") -> bytes | None:
        """The newest frame as JPEG, corrected and encoded here on the HTTP thread.

        These are the slow parts, and only a browser asking for a picture should pay for
        them: they must not happen in the control loop.
        """
        latest = self._camera.latest() if self._camera is not None else None
        if latest is None:
            return None
        with self._lock:
            seq, jpeg = self._frames.get(which, (None, None))
            if latest.seq == seq:
                return jpeg
        view = self._views.get(which)
        jpeg = encode_jpeg(view(latest.image) if view is not None else latest.image)
        with self._lock:
            self._frames[which] = (latest.seq, jpeg)
        return jpeg
