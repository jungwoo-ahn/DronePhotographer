"""Localhost HTTP server for the ground-station UI.

Stdlib only -- the laptop should need `pip install mavsdk opencv-python` and nothing
more to fly. Binds to 127.0.0.1 by default: this page can stop the aircraft, so it is
not something to expose on the drone's own wireless network by accident.

The only state-changing endpoints are the e-stop and its explicit release. There is
deliberately no "set velocity" route -- the UI observes the chain of order, it is not
another rung in it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"


def _handler(state, supervisor):
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
                return self._send(json.dumps(state.snapshot()).encode(), "application/json")
            if path == "/api/frame.jpg":
                jpeg = state.frame()
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
            self._send(b"not found", "text/plain", code=404)
    return H


class UiServer:
    """Serves the panel on a daemon thread; never blocks the control loop."""

    def __init__(self, state, supervisor, host: str = "127.0.0.1", port: int = 8080):
        self.state, self.supervisor = state, supervisor
        self.host, self.port = host, port
        self._srv = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "UiServer":
        self._srv = ThreadingHTTPServer((self.host, self.port),
                                        _handler(self.state, self.supervisor))
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever, name="ui",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
