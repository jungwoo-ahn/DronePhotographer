"""Persistent Blender render server: load the scene ONCE, then render camera poses on
demand. Eliminates the per-frame scene reload (~23s: BVH build + GPU upload + process
start) that `blender_render_pose.py` pays every frame — only the camera moves between
requests, so steady renders are ~1-2s. ~10-15x faster for multi-step rollouts.

File IPC (comm_dir should be on local /tmp for low latency):
  - server writes  <comm>/ready          once the scene is loaded
  - client writes  <comm>/request.json   {position, forward, up, output_image}  (atomic rename)
  - server renders, writes <comm>/response.json  {ok, image} | {ok: false, error}
  - client writes  <comm>/stop           to end the loop

Run (inside Blender):
  blender/blender -b -P scripts/blender_render_server.py -- --run_info_path <ri> --comm_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drones.blender_drone import BlenderDrone


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_info_path", required=True)
    p.add_argument("--comm_dir", required=True)
    split = argv.index("--") + 1 if "--" in argv else len(argv)
    return p.parse_args(argv[split:])


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    comm = Path(args.comm_dir)

    drone = BlenderDrone.from_run_info(args.run_info_path)   # scene + object loaded ONCE
    # Reuse Cycles' synced scene / BVH / GPU buffers across renders: a camera-only change
    # then skips the per-render geometry re-sync (~14s), so steady frames drop to ~1-3s.
    drone.scene.render.use_persistent_data = True
    (comm / "ready").write_text("1")
    print("RENDER_SERVER_READY", flush=True)

    req, resp, stop = comm / "request.json", comm / "response.json", comm / "stop"
    while True:
        if stop.exists():
            break
        if not req.exists():
            time.sleep(0.03)
            continue
        try:
            r = json.loads(req.read_text())
        except Exception:
            time.sleep(0.02)   # mid-write; retry
            continue
        req.unlink(missing_ok=True)
        try:
            drone.set_pose(position=r["position"], forward=r["forward"], up=r["up"])
            out = drone.render_rgb(r["output_image"])
            payload = {"ok": True, "image": str(out)}
        except Exception as e:   # keep the server alive across a bad frame
            payload = {"ok": False, "error": repr(e)}
        tmp = resp.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(resp)        # atomic — client never reads a partial response


if __name__ == "__main__":
    raw = list(getattr(bpy.app, "argv", sys.argv))
    if "--" not in raw and raw and raw[0].endswith(".py"):
        raw = raw[1:]
    main(raw)
