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


class PolicyClient:
    name = "policy"

    def __init__(self, url: str = "http://127.0.0.1:8777/act", timeout_s: float = 10.0,
                 session=None):
        self.url = url
        self.timeout_s = float(timeout_s)
        self._session = session
        self._seq = 0
        self.failures = 0
        self.calls = 0
        self.last_error = None

    def _sess(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def request_actions(self, frame_jpeg: bytes | None, goal: dict,
                        timeout_s: float | None = None) -> list | None:
        """One forward pass. The sequence of raw actions, or `None` -- never raises.

        An empty sequence is a failure too: that is the server saying it could not answer,
        and `last_error` keeps what it said.
        """
        self._seq += 1
        payload = {
            "seq": self._seq,
            "goal": goal,
            "image_b64": base64.b64encode(frame_jpeg).decode() if frame_jpeg else None,
        }
        self.calls += 1
        try:
            r = self._sess().post(self.url, json=payload, timeout=timeout_s or self.timeout_s)
            r.raise_for_status()
            reply = r.json()
            actions = reply.get("actions")
            if not actions:
                raise ValueError(reply.get("error") or "the server returned no actions")
            self.last_error = None
            return actions
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return None
