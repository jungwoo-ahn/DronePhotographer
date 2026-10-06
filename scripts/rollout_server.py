"""Persistent Blender render+score server for closed-loop policy rollout.

Loads the scene ONCE, then serves pose -> (rendered frame + achieved 8-key shot
profile) over a file handshake. Cycles+OptiX GPU (honors CUDA_VISIBLE_DEVICES, so
it stays off a forbidden GPU; EEVEE does not). The achieved profile is the same
v7 Stage-3 scorer the training goals come from (mesh-tight bbox + cam->obj az/el),
so goal-distance in the rollout is apples-to-apples with the conditioning goal.

  blender --background --python rollout_server.py -- \
      --data_json ... --v6_json ... --assets_root ... --ctl_dir ... --samples 16
"""
import sys, json, time, math, argparse, tempfile
from pathlib import Path
import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/DronePhotographer")
JW = Path("/home/nas5/jungwooahn/projects/DronePhotographer/scripts")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(JW))
import v7_sample_pairs_smoke as smoke
from src.scoring.bbox_control import compute_v5_scores

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--data_json", required=True)
ap.add_argument("--v6_json", required=True)
ap.add_argument("--assets_root", default=str(REPO))
ap.add_argument("--ctl_dir", required=True)
ap.add_argument("--res", nargs=2, type=int, default=[1024, 768])
ap.add_argument("--samples", type=int, default=16)
a = ap.parse_args(argv)
W, H = int(a.res[0]), int(a.res[1])
ctl = Path(a.ctl_dir); ctl.mkdir(parents=True, exist_ok=True)


def euler_xyz(rot):
    rx, ry, rz = [float(v) for v in rot]
    cx, sx = math.cos(rx), math.sin(rx); cy, sy = math.cos(ry), math.sin(ry); cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


stage1 = json.loads(Path(a.data_json).read_text())
idx = int(stage1.get("placement_idx", 0))
v6doc = json.loads(Path(a.v6_json).read_text())
chosen = v6doc["placements"][idx]
placement = {
    "name": stage1["placement"], "scene_file": stage1["scene_file"], "object_file": stage1["object_file"],
    "scene_scale": float(v6doc.get("scene_scale", 1.0)),
    "position": np.asarray([float(v) for v in chosen["position"]], dtype=np.float64),
    "rotation": [float(v) for v in chosen.get("rotation", [0, 0, 0])], "scale": float(chosen.get("scale", 1.0)),
}
O = np.asarray(stage1["subject_center"], dtype=np.float64)
R_obj_T = euler_xyz(chosen.get("rotation", [0, 0, 0])).T
elev_sign = -1 if str(stage1.get("stage3_elev_sign", "neg")) == "neg" else 1

# The models were retrained on SUBJECT-frame bearing (subject_bearing_deg), not world azimuth.
# Emit it so the sim goal space matches training: bearing = (front_az + yaw - azimuth) % 360,
# yaw=0 (our data.json has no placement_yaw_deg). Computed inline from the facing-map JSON —
# `import src.policy...` fails inside Blender's Python, so just read the file + do arithmetic.
import json as _json
_fm = _json.loads((Path(a.assets_root) / "configs" / "policy" / "facing_map_final.json").read_text())
_front_az = (_fm.get(Path(stage1["object_file"]).stem) or {}).get("front_az")

import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
meta = smoke.setup_blender_scene(placement, Path(a.assets_root))
cam = smoke.configure_renderer(W, H, int(a.samples), focal_length=24.0,
                               sensor_width=12.8, sensor_height=9.6, sky_strength=0.1)
# Cycles+OptiX GPU (pinned by CUDA_VISIBLE_DEVICES); EEVEE would ignore the pin.
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices:
    d.use = (d.type != "CPU")
scene.cycles.device = "GPU"; scene.render.use_persistent_data = True
frame_bounds = smoke._get_camera_frame_bounds(scene, cam)
in_frame_check = smoke.make_in_frame_check(meta["subject_verts_world"], frame_bounds, (W, H))


def score(pos, fwd, up):
    in_frame, occ, bbox = in_frame_check(pos, fwd, up)
    d = R_obj_T @ (O - np.asarray(pos, dtype=np.float64)); n = float(np.linalg.norm(d))
    u = d / n if n > 1e-9 else np.zeros(3)
    az = float(math.degrees(math.atan2(float(u[1]), float(u[0])))) % 360.0
    el = float(elev_sign * math.degrees(math.asin(float(np.clip(u[2], -1, 1)))))
    v5 = compute_v5_scores(image_width=W, image_height=H,
                           bbox_full=tuple(bbox) if bbox is not None else None, azimuth_deg=az, elevation_deg=el)
    out = {k: int(v) for k, v in v5.items()}
    if _front_az is not None:                          # world azimuth -> subject-frame bearing
        out["subject_bearing_deg"] = int(round((float(_front_az) - az) % 360.0))
    out["subject_distance_m"] = round(n, 3)      # for agents that reason about move scale
    return out, bool(in_frame), float(occ)


def scene_scout(max_objects=24):
    """Geometric scene summary for PhotoFlow's Director (paper 4.1: object names, bboxes,
    centers, scene extents). Written once at load; cheap, and nothing else reads it."""
    objs, lo, hi = [], None, None
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not len(ob.data.vertices):
            continue
        a = np.array([list(ob.matrix_world @ Vector(c)) for c in ob.bound_box], dtype=np.float64)
        bmin, bmax = a.min(0), a.max(0)
        lo = bmin if lo is None else np.minimum(lo, bmin)
        hi = bmax if hi is None else np.maximum(hi, bmax)
        dims = bmax - bmin
        objs.append({"name": ob.name, "center": [round(float(v), 2) for v in (bmin + bmax) / 2.0],
                     "size": [round(float(v), 2) for v in dims],
                     "volume": float(dims[0] * dims[1] * dims[2])})
    objs.sort(key=lambda o: -o["volume"])                 # biggest first = the ones worth naming
    for o in objs:
        o.pop("volume")
    ext = {"min": [round(float(v), 2) for v in lo], "max": [round(float(v), 2) for v in hi]} if lo is not None else {}
    return {"scene": stage1["placement"], "scene_scale": placement["scene_scale"],
            "extents": ext, "subject": {"names": stage1.get("subject_names", []),
                                        "center": [round(float(v), 3) for v in O],
                                        "height": stage1.get("subject_height")},
            "objects": objs[:max_objects], "n_objects": len(objs),
            "camera": {"focal_mm": 24.0, "sensor_mm": [12.8, 9.6], "resolution": [W, H],
                       "hfov_deg": 29.86, "vfov_deg": 22.62}}


try:
    (ctl / "scene.json").write_text(json.dumps(scene_scout(), indent=1))
except Exception as e:                                    # a scout failure must not kill rollouts
    print(f"scene_scout failed: {e}", flush=True)

(ctl / "ready.flag").write_text("ok")
print("server ready", flush=True)
while True:
    if (ctl / "stop.flag").exists():
        break
    req = ctl / "req.json"
    if req.exists():
        r = json.loads(req.read_text()); t = int(r["t"]); p = r["pose"]
        cam.matrix_world = smoke._camera_matrix_from_forward_up(p["pos"], p["forward"], p["up"])
        bpy.context.view_layer.update()
        img = ctl / f"frame_{t:03d}.jpg"; scene.render.filepath = str(img)
        bpy.ops.render.render(write_still=True)
        achieved, in_frame, occ = score(p["pos"], p["forward"], p["up"])
        achieved["in_frame"] = bool(in_frame)       # agents that read `achieved` need this too
        tmp = ctl / f"resp_{t:03d}.json.tmp"
        tmp.write_text(json.dumps({"image": str(img), "achieved": achieved, "in_frame": in_frame, "occ": occ}))
        tmp.rename(ctl / f"resp_{t:03d}.json")
        req.unlink()
    else:
        time.sleep(0.05)
print("server stop", flush=True)
