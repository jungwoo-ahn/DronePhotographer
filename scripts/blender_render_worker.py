"""Persistent Blender render worker (for fast RL rollouts; AutoPhoto baseline).

Run inside Blender, bound to one scene/object for its lifetime:

    blender -b -P scripts/blender_render_worker.py -- --run_info_path RI [--engine BLENDER_EEVEE_NEXT] [--samples 16]

Loads the scene + object ONCE (via BlenderDrone, which applies scene_scale + the
full object transform from the extended run_info), then loops over JSON commands on
stdin and renders on demand — avoiding the ~2-3s per-frame scene-load of the
subprocess-per-frame path. Protocol (one JSON object per line):

    {"position":[x,y,z], "forward":[x,y,z], "up":[x,y,z], "out":"/abs/frame.png"}
        -> renders that pose, prints {"ok": true, "path": "..."}
    {"cmd":"quit"}  -> exits

Switching the engine to EEVEE + low samples (flags) makes RL-scale rollouts
tractable. (Run on the render machine; no Blender binary in CI.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy  # noqa: F401  (provided by Blender)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drones.blender_drone import BlenderDrone


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--run_info_path", required=True)
    p.add_argument("--engine", default=None, help="e.g. BLENDER_EEVEE_NEXT for fast rollouts")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--cycles-gpu", action="store_true",
                   help="Cycles on GPU (OptiX). Unlike EEVEE, Cycles honors CUDA_VISIBLE_DEVICES, "
                        "so this is the only way to keep a headless render OFF a forbidden GPU.")
    i = argv.index("--") + 1 if "--" in argv else len(argv)
    return p.parse_args(argv[i:])


def _enable_cycles_gpu(scene) -> str:
    """Set Cycles + OptiX GPU. CUDA_VISIBLE_DEVICES masks which physical GPU(s)
    OptiX sees, so the render lands on the intended (non-forbidden) GPU. Returns
    a short status string for the ready handshake."""
    import bpy

    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    on = [d.name for d in prefs.devices if d.type != "CPU"]
    for d in prefs.devices:
        d.use = (d.type != "CPU")
    scene.cycles.device = "GPU"
    # Reuse the OptiX BVH across renders in the same scene — the BVH build on
    # heavy scenes dominates the FIRST render (can exceed a minute); persistent
    # data makes every subsequent render in that scene fast.
    scene.render.use_persistent_data = True
    return f"cycles/optix devices: {on}"


def main(argv) -> None:
    args = _parse(argv)
    drone = BlenderDrone.from_run_info(args.run_info_path)
    scene = drone.scene
    status = None
    if args.cycles_gpu:
        try:
            status = _enable_cycles_gpu(scene)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"warn": f"cycles-gpu setup failed: {e}"}), flush=True)
    elif args.engine:                     # speed: EEVEE instead of Cycles for RL
        try:
            scene.render.engine = args.engine
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"warn": f"engine {args.engine} failed: {e}"}), flush=True)
    if args.samples is not None:
        if hasattr(scene, "cycles"):
            scene.cycles.samples = args.samples
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = args.samples
    if status:
        print(json.dumps({"info": status}), flush=True)

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad json"}), flush=True)
            continue
        if cmd.get("cmd") == "quit":
            break
        try:
            drone.set_pose(position=cmd["position"], forward=cmd["forward"], up=cmd["up"])
            out = drone.render_rgb(cmd["out"])
            print(json.dumps({"ok": True, "path": str(out)}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main(sys.argv)
