"""Build ~20 diverse, realizable eval goal profiles from held-out val frames.

The legacy `configs/inference/*.yaml` are MPC-era + fractional units and cannot be
used with the current raw-unit goal space. Instead we sample the policy's ACTUAL
training-goal distribution: the stored 8-key V5 `scores` of real held-out frames
(guaranteed realizable, mutually consistent, correct integer units). Farthest-point
sampling in normalized goal space picks a maximally-spread subset; each goal is
labeled by its composition for interpretability.

Writes `configs/eval/eval_goals.json`.

Run on login (CPU only):
  PYTHONPATH=. .venv/bin/python scripts/build_eval_goals.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.policy.common.goal_space import normalize_goal
from src.scoring.bbox_control import V5_SCORE_KEYS

REPO = Path(__file__).resolve().parents[1]


def read_val_scenes(path: str) -> set[str]:
    return {
        ln.strip() for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }


def is_clamped(s: dict) -> bool:
    return s.get("occupancy") == 0 and s.get("bbox_y_offset") == 0


def is_well_composed(s: dict, W: int, H: int, min_bif: int, occ_lo: int, occ_hi: int) -> bool:
    """A 'good' framing: subject mostly in-frame, sensibly sized, center on screen.

    Random poses produce many degenerate edges (subject ~90% out of frame, or a
    frame-filling crop that shows a sliver of a huge projection). Those are real
    but not 'good' goals, and farthest-point sampling would over-select them.
    """
    if s["body_in_frame_ratio"] < min_bif:
        return False
    if not (occ_lo <= s["occupancy"] <= occ_hi):
        return False
    cx, cy = s["object_center_x"], s["object_center_y"]
    return 0 <= cx <= W and 0 <= cy <= H


def label(s: dict, W: int, H: int) -> tuple[str, str]:
    occ, el = s["occupancy"], s["cam_to_obj_elevation_deg"]
    cx, cy, az = s["object_center_x"], s["object_center_y"], s["cam_to_obj_azimuth_deg"]
    size = "closeup" if occ >= 40 else ("medium" if occ >= 15 else "wide")
    angle = "highangle" if el <= -15 else ("lowangle" if el >= 15 else "eyelevel")
    fx = cx / W
    horiz = "centered" if abs(fx - 0.5) < 0.12 else ("left" if fx < 0.5 else "right")
    fy = cy / H
    vert = "" if abs(fy - 0.5) < 0.15 else ("_top" if fy < 0.5 else "_low")
    oct_names = ["frontR", "right", "backR", "back", "backL", "left", "frontL", "front"]
    az_name = oct_names[int(((az + 22.5) % 360) // 45)]
    name = f"{size}_{horiz}{vert}_{angle}_{az_name}"
    desc = (
        f"{size} shot, subject {'centered' if horiz == 'centered' else horiz + '-third'}"
        f"{' high in frame' if vert == '_top' else ' low in frame' if vert == '_low' else ''}, "
        f"{angle} (elevation {el}deg), azimuth {az}deg, occupancy {occ}%"
    )
    return name, desc


def farthest_point(X: np.ndarray, n: int, seed_idx: int) -> list[int]:
    chosen = [seed_idx]
    d = np.linalg.norm(X - X[seed_idx], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--val-names", default="configs/policy/val_scenes.txt")
    ap.add_argument("--data-root", default="data/trajectories")
    ap.add_argument("--out", default="configs/eval/eval_goals.json")
    ap.add_argument("--min-bif", type=int, default=85, help="min body_in_frame_ratio for a 'good' framing")
    ap.add_argument("--occ-min", type=int, default=8)
    ap.add_argument("--occ-max", type=int, default=80)
    ap.add_argument("--no-compose-filter", action="store_true", help="keep edge/off-frame profiles too")
    args = ap.parse_args()

    val = read_val_scenes(args.val_names)
    base = Path(args.data_root)
    cand: list[tuple] = []  # (norm_vec, scores, placement, frame_idx, W, H)
    n_placements = 0
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.split("__")[0] not in val:
            continue
        dj = d / "data.json"
        if not dj.exists():
            continue
        n_placements += 1
        doc = json.loads(dj.read_text())
        W, H = doc.get("render_width", 1024), doc.get("render_height", 768)
        for pair in (doc.get("render_records") or []):
            for rec in pair:
                s = rec.get("scores")
                if not s or is_clamped(s) or not rec.get("in_frame", True):
                    continue
                if not args.no_compose_filter and not is_well_composed(
                    s, W, H, args.min_bif, args.occ_min, args.occ_max
                ):
                    continue
                vec = normalize_goal(np.array([s[k] for k in V5_SCORE_KEYS], dtype=np.float32), V5_SCORE_KEYS)
                cand.append((vec, s, d.name, rec.get("frame_idx"), W, H))

    if len(cand) < args.n:
        raise SystemExit(f"only {len(cand)} candidate frames in {n_placements} val placements (need {args.n})")

    X = np.stack([c[0] for c in cand])
    seed = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))  # medoid -> stable, central first pick
    idx = farthest_point(X, args.n, seed)

    goals, seen = [], {}
    for i in idx:
        _, s, pl, fr, W, H = cand[i]
        name, desc = label(s, W, H)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        goals.append({
            "name": name, "description": desc,
            "profile": {k: int(s[k]) for k in V5_SCORE_KEYS},
            "source_placement": pl, "source_frame": fr,
        })

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(goals, indent=2))
    print(f"wrote {len(goals)} goals from {len(cand)} held-out candidate frames "
          f"({n_placements} placements) -> {out}")
    for g in goals:
        p = g["profile"]
        print(f"  {g['name']:46s} occ={p['occupancy']:3d} cx={p['object_center_x']:4d} "
              f"cy={p['object_center_y']:4d} az={p['cam_to_obj_azimuth_deg']:3d} "
              f"el={p['cam_to_obj_elevation_deg']:4d} bx={p['bbox_x_offset']:3d} by={p['bbox_y_offset']:3d}")


if __name__ == "__main__":
    main()
