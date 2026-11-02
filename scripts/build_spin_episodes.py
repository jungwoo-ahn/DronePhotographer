"""Benchmark v2 with every subject spun in place: does a policy read the subject, or the scene?

Every training placement left its subject unrotated (rotation 0 in all 2,081 placements checked),
so throughout training each asset stood in its scene at the same orientation, and "which side of
the subject the camera is on" could be read off the room as well as off the body. On 3DGS scenes,
where subjects are turned to face the capture, every shot-profile policy picks its orbit direction
at chance while I2A does not. This separates the two readings: same episodes, same camera poses,
same scenes, the subject turned by --yaw about its own vertical axis (the renderer's pivot: the
bottom centre at the placement position), and the goal profile recomputed for the turned subject
exactly as the training labels were made. A policy that reads the body is unaffected; one that
reads the scene steers for where the subject's front used to be.

Writes `<out>/episodes.json` (kind "posed_blender", replayed by closed_loop_eval.py
--episodes-from with the v7 render profile), turned geometry in `<out>/_geom` for
score_benchmark.py, and per-placement run_info files carrying the turned subject.

    .venv-analysis/bin/python scripts/build_spin_episodes.py --yaw 180 --out runs/bench_v2_spin180
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)

import numpy as np  # noqa: E402

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector  # noqa: E402
from src.common.reward import _geometry_distance, pose_to_geometry  # noqa: E402
from src.common.run_info import RENDER_PROFILES, write_run_info  # noqa: E402
from src.goal_authoring.pose_solver import PlacementGeometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{V12}/configs/eval_episodes_v2.json")
ap.add_argument("--geom", default=f"{V12}/runs/bench_v2/_geom")
ap.add_argument("--yaw", type=float, required=True, help="degrees, about the subject's vertical axis")
ap.add_argument("--out", required=True)
args = ap.parse_args()
OUT = Path(args.out) if Path(args.out).is_absolute() else Path(V12) / args.out
PROFILE_KEYS = ("occupancy", "body_in_frame_ratio", "cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg",
                "object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset",
                "top_cut_frac", "bot_cut_frac", "visible_frac", "head_in_frame")


def spun_run_info(placement: str, data_json: str) -> tuple[str, np.ndarray]:
    """The v2 rollout's run_info with the subject turned; returns (path, pivot xy)."""
    opts = {"render_device": "GPU", "blender_threads": 8, "persistent_data": True, **RENDER_PROFILES["v7"]}
    p = Path(write_run_info(placement, data_json, OUT / "_run_info", shared_root=SHARED,
                            resolution=(1024, 768), extra_options=opts))
    ri = json.loads(p.read_text())
    rot = [float(v) for v in ri["rotation_xyz_rad"]]
    if abs(rot[0]) > 1e-9 or abs(rot[1]) > 1e-9:
        raise SystemExit(f"{placement}: tilted placement, a z-spin about the foot is not exact")
    rot[2] += math.radians(args.yaw)
    ri["rotation_xyz_rad"] = rot
    p.write_text(json.dumps(ri, indent=1))
    opts_doc = ri["options"]
    pos = (opts_doc["object_position"] if isinstance(opts_doc, dict)
           else __import__("ast").literal_eval(opts_doc)["object_position"])
    return str(p), np.asarray(pos[:2], dtype=np.float64)


def spun_geometry(placement: str, pivot: np.ndarray) -> PlacementGeometry:
    src = PlacementGeometry.load(Path(args.geom) / f"{placement}.npz")
    a = math.radians(args.yaw)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    def turn(xyz):
        out = np.array(xyz, dtype=np.float64, copy=True)
        out[..., :2] = (out[..., :2] - pivot) @ R.T + pivot
        return out
    # The dump's own centre/foot/height, turned -- not recomputed from the (subsampled) vertices,
    # so --yaw 0 reproduces benchmark v2 exactly.
    v = turn(src.verts)
    g = PlacementGeometry(placement=src.placement, verts=v, frame_bounds=src.frame_bounds,
                          width=src.width, height=src.height, subject_center=turn(src.subject_center),
                          subject_foot=turn(src.subject_foot), subject_height=float(src.subject_height))
    (OUT / "_geom").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "_geom" / f"{placement}.npz", placement=np.array(placement), verts=v,
                        frame_bounds=np.array(src.frame_bounds), resolution=np.array([src.width, src.height]),
                        subject_center=g.subject_center, subject_foot=g.subject_foot,
                        subject_height=np.float64(g.subject_height))
    return g


def main() -> int:
    doc = json.loads(Path(args.manifest).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    eps, moved = [], []
    for ep in doc["episodes"]:
        pl = ep["placement"]
        if pl not in cache:
            ri, pivot = spun_run_info(pl, ep["data_json"])
            foot = PlacementGeometry.load(Path(args.geom) / f"{pl}.npz").subject_foot[:2]
            if np.linalg.norm(foot - pivot) > 0.05:
                raise SystemExit(f"{pl}: renderer pivot {pivot} is not the subject's foot {foot}")
            cache[pl] = (ri, spun_geometry(pl, pivot))
        ri, g = cache[pl]
        want = (ep["pair_idx"], ep["start_frame_idx"], ep["goal_frame_idx"])
        w = next(w for w in iter_goal_start_windows(ep["data_json"], chunk_size=int(doc.get("chunk_size", 8)),
                                                   max_per_pair=int(doc.get("max_per_pair", 2)))
                 if (w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) == want)
        pose = lambda f: {"position": [float(x) for x in f.camera_position],
                          "forward": [float(x) for x in f.camera_forward], "up": [float(x) for x in f.camera_up]}
        sp, gp = pose(w.start), pose(w.goal_frame)
        prof = g.profile(*(np.asarray(gp[k]) for k in ("position", "forward", "up")))
        if prof is None:
            raise SystemExit(f"{ep['key']}: the turned subject leaves the goal frame")
        raw = dict(w.goal_frame.raw)
        raw.update({k: float(prof[k]) for k in PROFILE_KEYS if k in prof})
        raw["placement_yaw_deg"] = float(args.yaw)
        obj = pl.split("__", 1)[1]
        gv = goal_vector(raw, DEFAULT_GOAL_KEYS, object_key=obj)
        if not np.isfinite(gv).all():
            raise SystemExit(f"{ep['key']}: goal vector not finite")
        bearing = float(gv[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
        moved.append(((bearing - float(ep["bearing"]) + 180) % 360) - 180)
        kw = {"subject_center": g.subject_center, "subject_height": g.subject_height}
        d0 = float(_geometry_distance(pose_to_geometry(sp["position"], sp["forward"], sp["up"], **kw),
                                      pose_to_geometry(gp["position"], gp["forward"], gp["up"], **kw)))
        eps.append({"key": ep["key"], "placement": pl, "scene": pl.split("__")[0], "object": obj,
                    "run_info": ri, "traj": ep["pair_idx"], "start_frame_idx": ep["start_frame_idx"],
                    "goal_frame_idx": ep["goal_frame_idx"], "start_pose": sp, "goal_pose": gp,
                    "subject_center": [float(x) for x in g.subject_center],
                    "subject_height": float(g.subject_height), "goal_raw": raw,
                    "goal_vec": [float(x) for x in gv], "goal_keys": list(DEFAULT_GOAL_KEYS),
                    "delta": ep["delta"], "n_chunks": ep.get("n_chunks"), "sector": sector8(bearing),
                    "bearing": bearing, "d_start": d0,
                    "start_image": w.start.image, "goal_image": w.goal_frame.image,
                    "original_bearing": float(ep["bearing"]), "original_d_start": float(ep["d_start"])})
    (OUT / "episodes.json").write_text(json.dumps({
        "_what": f"configs/eval_episodes_v2.json with every subject turned {args.yaw:g} deg in place; "
                 "camera poses unchanged, goal profiles recomputed. scripts/build_spin_episodes.py",
        "kind": "posed_blender", "render_profile": "v7", "chunk_size": int(doc.get("chunk_size", 8)),
        "subject_yaw_deg": args.yaw, "n_episodes": len(eps), "episodes": eps}, indent=1))
    m = np.array(moved)
    print(f"[spin] {len(eps)} episodes, {len(cache)} placements -> {OUT}/episodes.json; goal bearing "
          f"moved by median {np.median(m):+.1f} deg (spread {np.percentile(m, 5):+.1f}..{np.percentile(m, 95):+.1f}); "
          f"d_start mean {np.mean([e['d_start'] for e in eps]):.3f} (was "
          f"{np.mean([e['original_d_start'] for e in eps]):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
