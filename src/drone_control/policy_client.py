"""Laptop -> policy server over HTTP, through the SSH tunnel.

The server is a GPU box on the other end of a tunnel, so every call is assumed to be
slow and allowed to fail. This client therefore **never raises into the control loop**:
on timeout, connection error or garbage JSON it returns `None`, and the supervisor's
TTL turns repeated failures into a hover. A control loop that can be killed by a
`requests.exceptions.ReadTimeout` is a control loop that lands in a field.

Tunnel (from the laptop):

    ssh -N -L 8777:localhost:8777 <user>@<server>

then point `--policy-url` at http://127.0.0.1:8777/act
"""

from __future__ import annotations

import base64
import time

from .commands import VelocityCommand


class PolicyClient:
    name = "policy"

    def __init__(self, url: str = "http://127.0.0.1:8777/act", timeout_s: float = 0.8,
                 session=None):
        self.url = url
        #: Shorter than the command TTL on purpose: a request that outlives the command
        #: it would produce is worse than no request, because it occupies the slot where
        #: a fresh one could have gone.
        self.timeout_s = float(timeout_s)
        self._session = session
        self._seq = 0
        self.failures = 0
        self.calls = 0

    def _sess(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def request_command(self, frame_jpeg: bytes | None, state, goal: dict
                        ) -> VelocityCommand | None:
        """One round trip. Returns `None` on any failure -- never raises."""
        self._seq += 1
        payload = {
            "seq": self._seq,
            "goal": goal,
            "state": {"altitude": getattr(state, "altitude", 0.0),
                      "north": getattr(state, "north", 0.0),
                      "east": getattr(state, "east", 0.0)},
            "image_b64": base64.b64encode(frame_jpeg).decode() if frame_jpeg else None,
        }
        self.calls += 1
        try:
            r = self._sess().post(self.url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            return VelocityCommand.from_json(r.json(), source=self.name)
        except Exception:
            self.failures += 1
            return None

    def next_command(self, state, now: float) -> VelocityCommand | None:
        """`CommandSource` interface, for sources that carry their own frame/goal."""
        return self.request_command(None, state, {})
