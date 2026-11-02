"""Policy service -- runs on the GPU server, answers one velocity command per frame.

    python scripts/serve_policy.py --port 8777 --stub
    # laptop:
    ssh -N -L 8777:localhost:8777 <user>@<server>

Right now this is a STUB: it returns a slow left orbit so the laptop's control chain,
the SSH tunnel and the JSON contract can all be exercised before the cosmos3
checkpoint is consolidated. `--stub` is the default and the real path raises, so
nobody can mistake a stub reply for a model prediction.

Wiring the real policy in later means implementing `act()` only; the contract stays
`{vx, vy, vz, yaw_rate, ttl_s, note}` in OUR conventions (vy left+, vz up+, yaw_rate
left+ / counter-clockwise). The laptop converts to MAVSDK's frame -- the server must
never do that conversion itself, or it gets applied twice.
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubPolicy:
    """A gentle left orbit: forward with a counter-clockwise yaw."""
    name = "stub-orbit"

    def act(self, image_bytes, state, goal):
        return {"vx": 0.8, "vy": 0.0, "vz": 0.0, "yaw_rate": 12.0,
                "ttl_s": 1.0, "note": f"{self.name}: orbiting left"}


class CosmosPolicy:
    def __init__(self, ckpt):
        raise NotImplementedError(
            "The cosmos3 checkpoint is 8 FSDP .distcp shards and needs consolidating "
            "into a single state dict before it can be served. Run with --stub until "
            "that exists."
        )


def make_handler(policy):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                  # quiet; the control loop does the logging

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                img = req.get("image_b64")
                out = policy.act(base64.b64decode(img) if img else None,
                                 req.get("state") or {}, req.get("goal") or {})
                out["seq"] = req.get("seq", 0)
                body = json.dumps(out).encode()
                self.send_response(200)
            except Exception as e:
                # Answer with an explicit hover rather than a 500: the laptop treats a
                # failed call as "no opinion" and keeps the previous command until it
                # expires, whereas an explicit zero stops the drone immediately -- which
                # is what we want when the server itself is broken.
                body = json.dumps({"vx": 0, "vy": 0, "vz": 0, "yaw_rate": 0,
                                   "ttl_s": 1.0, "note": f"server error: {e}"}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--host", default="127.0.0.1", help="loopback: reach it over an SSH tunnel")
    p.add_argument("--stub", action="store_true", default=True)
    p.add_argument("--checkpoint", default=None, help="implies the real policy (not yet wired)")
    a = p.parse_args()
    policy = CosmosPolicy(a.checkpoint) if a.checkpoint else StubPolicy()
    srv = HTTPServer((a.host, a.port), make_handler(policy))
    print(f"[serve] {policy.name} on http://{a.host}:{a.port}/act")
    print(f"[serve] tunnel from the laptop:  ssh -N -L {a.port}:localhost:{a.port} <user>@<server>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
