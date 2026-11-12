"""Localhost HTTP server for the ground-station UI.

Stdlib only -- the laptop should need `pip install mavsdk opencv-python` and nothing
more to fly. Binds to 127.0.0.1 by default: this page can stop the aircraft, so it is
not something to expose on the drone's own wireless network by accident.

The state-changing endpoints are the e-stop, its explicit release, (when the pilot is
flying under approval) accepting or skipping the move the policy has proposed,
picking the shot the policy is asked for, and taking a photo (saved on the laptop). There is deliberately no "set velocity" route: the panel can refuse a move
or run the one on offer, but it cannot author one. See `approval.ApprovalGate`.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

STATIC = Path(__file__).parent / "static"


def _json(obj) -> bytes:
    """JSON a browser can read. Python writes infinity as `Infinity`, which `JSON.parse`
    refuses, and one offer that never expires once froze the whole panel that way."""
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(obj), allow_nan=False).encode()


def _handler(state, supervisor, gate=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass                        # the control loop owns the console

        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass                    # browser navigated away mid-write; harmless

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), "text/html")
            if path == "/app.js":
                return self._send((STATIC / "app.js").read_bytes(), "application/javascript")
            if path == "/style.css":
                return self._send((STATIC / "style.css").read_bytes(), "text/css")
            if path == "/api/state":
                return self._send(_json(state.snapshot()), "application/json")
            if path == "/api/shots":
                return self._send(_json(state.shots()), "application/json")
            if path == "/api/shot.jpg":
                query = parse_qs(self.path.partition("?")[2])
                saved = state.shot_image((query.get("id") or [""])[0])
                if saved is None or not saved.exists():
                    return self._send(b"not found", "text/plain", code=404)
                return self._send(saved.read_bytes(), "image/jpeg", cache=True)
            if path == "/api/photo.jpg":
                query = parse_qs(self.path.partition("?")[2])
                saved = state.photo_path((query.get("name") or [""])[0])
                if saved is None or not saved.exists():
                    return self._send(b"not found", "text/plain", code=404)
                return self._send(saved.read_bytes(), "image/jpeg", cache=True)
            if path in ("/api/frame.jpg", "/api/model_frame.jpg"):
                jpeg = state.frame("model" if "model" in path else "wide")
                if not jpeg:
                    return self._send(b"", "image/jpeg", code=204)
                return self._send(jpeg, "image/jpeg")
            self._send(b"not found", "text/plain", code=404)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/estop":
                supervisor.estop()
                return self._send(b'{"estop":true}', "application/json")
            if path == "/api/clear_estop":
                supervisor.clear_estop()
                return self._send(b'{"estop":false}', "application/json")
            # Accept and skip act on the gate, never on the drone: the most either can
            # do is release the move the policy already proposed, or refuse it.
            if path == "/api/accept" and gate is not None:
                # A latched stop outranks the operator's click: accepting here would
                # consume the move and write "flown" in the record for a move the
                # supervisor was never going to let out.
                ran = False if supervisor.stats.get("estopped") else gate.accept()
                return self._send(json.dumps({"ran": ran}).encode(), "application/json")
            if path == "/api/skip" and gate is not None:
                return self._send(json.dumps({"skipped": gate.skip()}).encode(),
                                  "application/json")
            # Saves a frame on the laptop; nothing reaches the aircraft or the camera.
            if path == "/api/photo":
                return self._send(_json({"taken": state.take_photo()}), "application/json")
            # A new shot changes the goal the policy is asked with, nothing else. The move on
            # offer was for the old one, so it is taken back; the operator still runs each move.
            if path == "/api/shot":
                query = parse_qs(self.path.partition("?")[2])
                chosen = state.select_shot((query.get("id") or [""])[0])
                if chosen and gate is not None:
                    gate.withdraw("withdrawn: new shot")
                return self._send(json.dumps({"chosen": chosen}).encode(), "application/json")
            self._send(b"not found", "text/plain", code=404)
    return H


class UiServer:
    """Serves the panel on a daemon thread; never blocks the control loop."""

    def __init__(self, state, supervisor, host: str = "127.0.0.1", port: int = 8080,
                 gate=None):
        self.state, self.supervisor, self.gate = state, supervisor, gate
        self.host, self.port = host, port
        self._srv = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "UiServer":
        self._srv = ThreadingHTTPServer((self.host, self.port),
                                        _handler(self.state, self.supervisor, self.gate))
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever, name="ui",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
