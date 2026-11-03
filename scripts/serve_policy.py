"""Policy service -- runs on the GPU server, answers one velocity command per frame.

    python scripts/serve_policy.py --port 8777 --stub
    # laptop:
    ssh -N -L 8777:localhost:8777 <user>@<server>

Without `--checkpoint` this is a STUB: a slow left orbit, so the laptop's control chain,
the SSH tunnel and the JSON contract can all be exercised on any machine. The real policy
starts through scripts/serve_cosmos_policy.sh, which sets up what it needs.

The contract is `{vx, vy, vz, yaw_rate, ttl_s, note}` in OUR conventions (vy left+, vz up+,
yaw_rate left+ / counter-clockwise), and the real policy adds tilt, shoot and its chunk. The
laptop converts to MAVSDK's frame -- the server must never do that conversion itself, or it
gets applied twice.
"""

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubPolicy:
    """A gentle left orbit: forward with a counter-clockwise yaw."""
    name = "stub-orbit"

    def act(self, image_bytes, state, goal):
        return {"vx": 0.8, "vy": 0.0, "vz": 0.0, "yaw_rate": 12.0,
                "ttl_s": 1.0, "note": f"{self.name}: orbiting left"}


class CosmosPolicy:
    """The cosmos3 camera policy. Start it with scripts/serve_cosmos_policy.sh.

    Each request needs the frame, `state.tilt_deg` and a complete `goal.profile`. Anything
    missing raises, which the handler turns into an explicit hover: a free-text brief or an
    unknown tilt would produce a confident answer to the wrong question.
    """
    name = "cosmos3-camera-policy"

    def __init__(self, ckpt, *, vae, output_dir, move_s, samples, num_steps):
        from cosmos_framework.inference.common.init import init_script

        init_script()
        from src.policy.common.action_to_drone import action_to_move
        from src.policy.cosmos3.policy import Cosmos3CameraPolicy

        self._to_move = action_to_move
        self.move_s = float(move_s)
        self.policy = Cosmos3CameraPolicy(ckpt, vae_path=vae, output_dir=output_dir,
                                          samples=samples, num_steps=num_steps)

    def act(self, image_bytes, state, goal):
        import numpy as np
        from PIL import Image

        if not image_bytes:
            raise ValueError("no camera frame")
        tilt = float(state["tilt_deg"])
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        chunk = self.policy.predict_chunk(image, goal["profile"])
        # Only the first action: the view has moved by the time the rest would fly.
        m = self._to_move(chunk[0], tilt, self.move_s)
        return {"vx": m.vx, "vy": m.vy, "vz": m.vz, "yaw_rate": m.yaw_rate,
                "tilt_rate": m.tilt_rate, "tilt_target": m.tilt_target,
                "tilt_limited": m.tilt_limited, "shoot": m.shoot, "ttl_s": self.move_s,
                "chunk": chunk.tolist(), "note": self.name}


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
    p.add_argument("--checkpoint", default=None, help="implies the real policy")
    p.add_argument("--vae", default=None, help="Wan2.2_VAE.pth the checkpoint trained with")
    p.add_argument("--output-dir", default="runs/serve_policy", dest="output_dir")
    p.add_argument("--move-seconds", type=float, default=1.5, dest="move_seconds",
                   help="how long one action flies; the renders have no clock, so this sets speed")
    p.add_argument("--samples", type=int, default=4, help="draws averaged per reply")
    p.add_argument("--num-steps", type=int, default=30, dest="num_steps", help="sampler steps")
    a = p.parse_args()
    if a.checkpoint:
        policy = CosmosPolicy(a.checkpoint, vae=a.vae, output_dir=a.output_dir,
                              move_s=a.move_seconds, samples=a.samples, num_steps=a.num_steps)
    else:
        policy = StubPolicy()
    srv = HTTPServer((a.host, a.port), make_handler(policy))
    # Flushed: under a log file these are the only sign that the model has finished loading.
    print(f"[serve] {policy.name} on http://{a.host}:{a.port}/act", flush=True)
    print(f"[serve] tunnel from the laptop:  ssh -N -L {a.port}:localhost:{a.port} <user>@<server>",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
