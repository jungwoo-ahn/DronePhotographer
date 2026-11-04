"""Policy service -- runs on the GPU server, one forward pass per request.

    python scripts/serve_policy.py --port 8777
    # laptop:
    ssh -N -L 8777:localhost:8777 <user>@<server>

This is the whole interface between the machines:

    POST /act  {"image_b64": <jpeg>, "goal": {"profile": {...}}}
          ->   {"actions": [[10 floats] x 8], "note": ...}
          or   {"actions": [], "error": ...}

An action is the policy's raw `[Δtranslation_cam(3), rot6d(6), shoot]`. The server knows
nothing about the aircraft: holding the sequence, flying it and asking the operator about
each action all belong to the laptop.

Without `--checkpoint` this is a STUB, a slow left orbit on any machine, for exercising the
tunnel and the laptop. The real policy starts through scripts/serve_cosmos_policy.sh.
"""

import argparse
import base64
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubPolicy:
    """A gentle left orbit: forward with a counter-clockwise turn, whatever it is shown."""
    name = "stub-orbit"

    def act(self, image_bytes, goal):
        turn = -math.radians(6.0)       # about the OpenCV camera's down axis: negative is left
        c, s = math.cos(turn), math.sin(turn)
        action = [0.0, 0.0, 0.4, c, 0.0, -s, 0.0, 1.0, 0.0, 0.0]
        return {"actions": [action] * 8, "note": self.name}


class CosmosPolicy:
    """The cosmos3 camera policy. Start it with scripts/serve_cosmos_policy.sh."""
    name = "cosmos3-camera-policy"

    def __init__(self, ckpt, *, vae, output_dir, samples, num_steps):
        from cosmos_framework.inference.common.init import init_script

        init_script()
        from src.policy.cosmos3.policy import Cosmos3CameraPolicy

        self.policy = Cosmos3CameraPolicy(ckpt, vae_path=vae, output_dir=output_dir,
                                          samples=samples, num_steps=num_steps)

    def act(self, image_bytes, goal):
        import numpy as np
        from PIL import Image

        if not image_bytes:
            raise ValueError("no camera frame")
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        chunk = self.policy.predict_chunk(image, goal["profile"])
        return {"actions": chunk.tolist(), "note": self.name}


def make_handler(policy):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                  # quiet; the control loop does the logging

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                img = req.get("image_b64")
                out = policy.act(base64.b64decode(img) if img else None, req.get("goal") or {})
                out["seq"] = req.get("seq", 0)
            except Exception as e:
                # Says what went wrong, and offers nothing to fly.
                out = {"actions": [], "error": f"{type(e).__name__}: {e}"}
            body = json.dumps(out).encode()
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
    p.add_argument("--samples", type=int, default=4, help="draws averaged per reply")
    p.add_argument("--num-steps", type=int, default=30, dest="num_steps", help="sampler steps")
    a = p.parse_args()
    if a.checkpoint:
        policy = CosmosPolicy(a.checkpoint, vae=a.vae, output_dir=a.output_dir,
                              samples=a.samples, num_steps=a.num_steps)
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
