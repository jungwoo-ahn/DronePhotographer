"""Extract subject world-space vertices + camera frame bounds for a validation
sample — using the EXACT `BlenderDrone.from_run_info` render setup so the geometry
(scene_scale + placement transform) matches the dataset render pixel-for-pixel.

Output feeds `src.scoring.projection.score_pose`, so a rollout's achieved shot
profile is computed the same way the v7 training goals were (validated: stored
scores reproduce 1312/1312; world-frame az/el 100%). Per-frame scoring then needs
no Blender — only this one-time extraction does.

Run via:
  blender/blender -b -P scripts/extract_subject_geom.py -- \
      --run_info_path <run_info.json> --output_json <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drones.blender_drone import BlenderDrone

MAX_VERTS = 5000


def camera_frame_bounds(scene, cam_obj):
    """Frame corners normalized to z=1 -> (min_x, max_x, min_y, max_y).

    Same as scripts/compute_mesh_tight_bbox.py:get_camera_frame_bounds.
    """
    frame = cam_obj.data.view_frame(scene=scene)
    view_z = abs(frame[0].z) or 1.0
    xs = [f.x / view_z for f in frame]
    ys = [f.y / view_z for f in frame]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def subject_meshes():
    """Mesh descendants of the AutoPlacedRoot empty (the placed subject)."""
    roots = [o for o in bpy.data.objects if o.name.startswith("AutoPlacedRoot")]
    meshes = []
    for root in roots:
        for o in [root, *root.children_recursive]:
            if o.type == "MESH":
                meshes.append(o)
    return meshes


def world_vertices(meshes) -> np.ndarray:
    parts = []
    for m in meshes:
        local = np.array([v.co[:] for v in m.data.vertices], dtype=np.float64)
        if local.shape[0] == 0:
            continue
        mw = np.array(m.matrix_world, dtype=np.float64)
        homog = np.concatenate([local, np.ones((local.shape[0], 1))], axis=1)
        parts.append((homog @ mw.T)[:, :3])
    verts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
    if verts.shape[0] > MAX_VERTS:
        idx = np.linspace(0, verts.shape[0] - 1, MAX_VERTS).astype(np.int64)
        verts = verts[idx]
    return verts


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_info_path", required=True)
    ap.add_argument("--output_json", required=True)
    split = argv.index("--") + 1 if "--" in argv else 0
    args = ap.parse_args(argv[split:])

    drone = BlenderDrone.from_run_info(args.run_info_path)
    meshes = subject_meshes()
    if not meshes:
        raise RuntimeError("no subject meshes (AutoPlacedRoot descendants) found")
    verts = world_vertices(meshes)
    fb = camera_frame_bounds(drone.scene, drone.camera)
    out = {
        "verts_world": verts.tolist(),
        "frame_bounds": list(fb),
        "render_width": int(drone.scene.render.resolution_x),
        "render_height": int(drone.scene.render.resolution_y),
        "n_verts": int(verts.shape[0]),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out))
    print(f"extracted {verts.shape[0]} verts | frame_bounds={fb} | "
          f"res={out['render_width']}x{out['render_height']}")


if __name__ == "__main__":
    raw = list(getattr(bpy.app, "argv", sys.argv))
    if "--" not in raw and raw and raw[0].endswith(".py"):
        raw = raw[1:]
    main(raw)
