"""Can the orbit direction be read from FRAMING alone, without the bearing? (the 3DGS investigation)

The profile-conditioned policies choose which way to go around the subject without using the goal
bearing (`build_bearing_probe.py`: change only the bearing, the policy still goes to the original
goal). This shows where the direction comes from instead. A classifier is trained on training
windows (v7 trajectories, train scenes) to predict the sign of the needed orbit from the start
frame's and the goal's framing keys only -- occupancy, body-in-frame, elevation, subject centre,
box offsets, cuts; NO bearing -- and is then applied to benchmark v2 episodes (same trajectory
generator, held-out scenes) and to the 3DGS episodes.

Measured 2026-10-19: 5-fold CV on training windows 0.996; benchmark v2 0.99; 3DGS 0.48 (chance).
The policies show the same split (first-chunk direction 100% on v2, 55-61% on 3DGS).

    .venv-analysis/bin/python scripts/gs_investigation/direction_shortcut.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=f"{V12}/data/trajectories/v7_stage2_renders_lookat075")
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--n-placements", type=int, default=400, help="train placements sampled for windows")
ap.add_argument("--v2", default=f"{V12}/configs/eval_episodes_v2.json")
ap.add_argument("--gs", default=f"{V12}/configs/eval_episodes_gs_v1.json")
ap.add_argument("--gs-geom", default=f"{V12}/runs/bench_gs/_geom")
ap.add_argument("--min-need", type=float, default=15.0, help="deg; smaller orbits are not scored")
args = ap.parse_args()

KEYS = ["occupancy", "body_in_frame_ratio", "cam_to_obj_elevation_deg", "object_center_x",
        "object_center_y", "bbox_x_offset", "bbox_y_offset", "top_cut_frac", "bot_cut_frac",
        "visible_frac"]


def wrap(a: float) -> float:
    return (a + 180) % 360 - 180


def need_deg(start_pos, goal_pos, center) -> float:
    """Position-angle change around the subject from start to goal (the shorter way)."""
    ang = lambda p: math.degrees(math.atan2(p[1] - center[1], p[0] - center[0]))
    return wrap(ang(goal_pos) - ang(start_pos))


def window_features(w) -> tuple[list[float], float] | None:
    s, g = w.start.raw, w.goal_frame.raw
    if not isinstance(s, dict) or not all(k in s for k in KEYS) or not all(k in g for k in KEYS):
        return None
    n = need_deg(np.asarray(w.start.camera_position), np.asarray(w.goal_frame.camera_position),
                 np.asarray(w.goal_frame.subject_center))
    return [float(s[k]) for k in KEYS] + [float(g[k]) for k in KEYS], n


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    from src.common.annotations import iter_goal_start_windows
    from src.goal_authoring.pose_solver import PlacementGeometry

    val = set(json.loads(Path(args.val_scenes).read_text())["scenes"])
    dirs = sorted(d for d in os.listdir(args.root) if d.split("__")[0] not in val)
    random.Random(0).shuffle(dirs)
    X, Y = [], []
    for d in dirs[:args.n_placements]:
        p = os.path.join(args.root, d, "data.json")
        if not os.path.exists(p):
            continue
        try:
            for w in iter_goal_start_windows(p, chunk_size=8, max_per_pair=2):
                f = window_features(w)
                if f and abs(f[1]) > args.min_need:
                    X.append(f[0]); Y.append(f[1] > 0)
        except Exception:  # noqa: BLE001
            continue
    X, Y = np.array(X), np.array(Y)
    clf = HistGradientBoostingClassifier(max_iter=300)
    print(f"training windows: {len(Y)}; 5-fold CV accuracy (framing only, no bearing): "
          f"{cross_val_score(clf, X, Y, cv=5).mean():.3f}")
    clf.fit(X, Y)

    # benchmark v2: the stored start/goal labels of each episode's own window
    Xv, Yv = [], []
    for e in json.loads(Path(args.v2).read_text())["episodes"]:
        w = next((w for w in iter_goal_start_windows(e["data_json"], chunk_size=8, max_per_pair=2)
                  if (w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx)
                  == (e["pair_idx"], e["start_frame_idx"], e["goal_frame_idx"])), None)
        f = window_features(w) if w is not None else None
        if f and abs(f[1]) > args.min_need:
            Xv.append(f[0]); Yv.append(f[1] > 0)
    print(f"benchmark v2: accuracy {np.mean(clf.predict(np.array(Xv)) == np.array(Yv)):.2f} (n={len(Yv)})")

    # 3DGS: start labels from the placement geometry (the same scorer), goal labels from the manifest
    Xg, Yg, geoms = [], [], {}
    for e in json.loads(Path(args.gs).read_text())["episodes"]:
        g = geoms.setdefault(e["placement"], PlacementGeometry.load(Path(args.gs_geom) / f"{e['placement']}.npz"))
        sp = e["start_pose"]
        ps = g.profile(*(np.asarray(sp[k], dtype=np.float64) for k in ("position", "forward", "up")))
        if ps is None or not all(k in ps for k in KEYS):
            continue
        n = need_deg(np.asarray(sp["position"]), np.asarray(e["goal_pose"]["position"]),
                     np.asarray(e["subject_center"]))
        if abs(n) > args.min_need:
            Xg.append([float(ps[k]) for k in KEYS] + [float(e["goal_raw"][k]) for k in KEYS])
            Yg.append(n > 0)
    print(f"3DGS: accuracy {np.mean(clf.predict(np.array(Xg)) == np.array(Yg)):.2f} (n={len(Yg)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
