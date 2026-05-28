#!/usr/bin/env python3
"""V7 Stage 3 — V5 per-frame scoring (CPU-only, no detector).

Reads each placement's ``data.json`` (which carries mesh-projected
``bbox_xyxy_full`` per frame, populated by ``v7_stage2_render.py`` or
``v7_stage2_backfill_bbox.py``), then computes the **8 V5 integer keys**
per frame via ``src/scoring/bbox_control.py:compute_v5_scores``:

  - occupancy, body_in_frame_ratio
  - cam_to_obj_azimuth_deg, cam_to_obj_elevation_deg
  - object_center_x, object_center_y
  - bbox_x_offset, bbox_y_offset

cam→obj azim/elev follow the v2 convention (CLAUDE.md): elev = -90 means
camera directly above subject (cam→obj points straight down).

Other score families (7 rule-based, 6 camera-3D, 8 subject-aware) are
intentionally not computed — V5 is the schema the policy consumes.

Usage:
    python3 scripts/v7_stage3_score.py \\
        --out-dir outputs/v7_stage2_smoke_diverse \\
        --assignment-file splits/v7_stage2_assignments.json --side jungwooahn \\
        --resume
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scoring.bbox_control import compute_v5_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="outputs/v7_stage2_renders")
    p.add_argument(
        "--placements-v6-dir",
        default="data/vlm_object_placing_v6_260428_061326",
        help="v6 placement JSONs (needed for object rotation per placement).",
    )
    p.add_argument("--assignment-file",
                   default="splits/v7_stage2_assignments.json")
    p.add_argument("--side", default="jungwooahn")
    p.add_argument("--pilot-count", type=int, default=0)
    p.add_argument("--only-placement", default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip placements whose scored.flag exists.")
    p.add_argument(
        "--require-bbox", action="store_true", default=True,
        help="Skip placements lacking bbox_xyxy_full (Stage 2 not yet backfilled).",
    )
    p.add_argument(
        "--elev-sign", choices=["pos", "neg"], default="pos",
        help="Sign for elev = SIGN * arcsin(unit_z_obj). 'pos' matches v2 "
             "(cam above → elev<0). Override if spot-check disagrees.",
    )
    return p.parse_args()


def load_assignment(path: Path, side: str) -> list[str]:
    doc = json.loads(path.read_text())
    return list(doc["sides"][side]["placements"])


def load_v6_placement(v6_path: Path, placement_idx: int) -> dict:
    doc = json.loads(v6_path.read_text())
    chosen = doc["placements"][placement_idx]
    return {
        "rotation": [float(v) for v in chosen.get("rotation", [0.0, 0.0, 0.0])],
        "scale": float(chosen.get("scale", 1.0)),
    }


def euler_xyz_rad_to_matrix(rot: list[float]) -> np.ndarray:
    """Intrinsic XYZ Euler (Blender default) → 3×3 rotation matrix.

    R = Rx · Ry · Rz applied to a column vector: v_world = R · v_local.
    Matches Blender's `Euler((rx, ry, rz), 'XYZ').to_matrix()`.
    """
    rx, ry, rz = (float(v) for v in rot)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def score_frame(
    bbox: list[float] | None,
    W: int,
    H: int,
    O: np.ndarray,
    cam_pos: np.ndarray,
    R_obj_T: np.ndarray,   # R^T applied to world vectors → object-local
    elev_sign: int,
) -> dict:
    """Compute 8 V5 integer keys for one frame."""
    # cam→obj direction in object-local frame (v2 convention)
    d_world = O - cam_pos
    d_obj = R_obj_T @ d_world
    norm = float(np.linalg.norm(d_obj))
    if norm > 1e-9:
        unit = d_obj / norm
    else:
        unit = np.zeros(3)
    azim_deg = float(math.degrees(math.atan2(float(unit[1]), float(unit[0])))) % 360.0
    z_clamped = float(np.clip(unit[2], -1.0, 1.0))
    elev_deg = float(elev_sign * math.degrees(math.asin(z_clamped)))

    v5 = compute_v5_scores(
        image_width=W, image_height=H,
        bbox_full=tuple(bbox) if bbox is not None else None,
        azimuth_deg=azim_deg,
        elevation_deg=elev_deg,
    )
    return {k: int(v) for k, v in v5.items()}


def score_placement(
    name: str,
    out_dir: Path,
    v6_dir: Path,
    args: argparse.Namespace,
) -> dict:
    placement_dir = out_dir / name
    data_path = placement_dir / "data.json"
    if not data_path.exists():
        return {"name": name, "status": "missing"}
    if args.resume and (placement_dir / "scored.flag").exists():
        return {"name": name, "status": "skip"}

    data = json.loads(data_path.read_text())
    render_records = data.get("render_records") or []
    accepted_pairs = data.get("accepted_pairs") or []

    if args.require_bbox:
        has_bbox = False
        for pair_recs in render_records:
            for rec in pair_recs:
                if "bbox_xyxy_full" in rec:
                    has_bbox = True
                    break
            if has_bbox:
                break
        if not has_bbox:
            return {"name": name, "status": "no_bbox"}

    W = int(data.get("render_width") or 0)
    H = int(data.get("render_height") or 0)
    if W <= 0 or H <= 0:
        return {"name": name, "status": "no_resolution"}

    O = np.asarray(data["subject_center"], dtype=np.float64)
    v6 = load_v6_placement(
        v6_dir / f"{name}.json",
        int(data.get("placement_idx", 0)),
    )
    R_obj = euler_xyz_rad_to_matrix(v6["rotation"])
    R_obj_T = R_obj.T

    elev_sign = -1 if args.elev_sign == "neg" else 1

    n_scored = 0
    t_start = time.time()
    for pair_idx, pair_recs in enumerate(render_records):
        if pair_idx >= len(accepted_pairs):
            continue
        traj = accepted_pairs[pair_idx].get("trajectory_32f") or []
        for rec in pair_recs:
            fi = int(rec.get("frame_idx", -1))
            if fi < 0 or fi >= len(traj):
                continue
            bbox = rec.get("bbox_xyxy_full")
            cam_pos = np.asarray(traj[fi]["pos"], dtype=np.float64)
            rec["scores"] = score_frame(
                bbox, W, H, O, cam_pos, R_obj_T, elev_sign,
            )
            n_scored += 1

    data["render_records"] = render_records
    data["stage3_n_frames"] = int(n_scored)
    data["stage3_time_s"] = float(time.time() - t_start)
    data["stage3_elev_sign"] = args.elev_sign
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (placement_dir / "scored.flag").write_text(
        f"frames={n_scored}  time={time.time()-t_start:.3f}s  "
        f"elev_sign={args.elev_sign}\n",
        encoding="utf-8",
    )
    return {"name": name, "status": "ok", "n_frames": n_scored,
            "elapsed": time.time() - t_start}


def main() -> int:
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    v6_dir = (REPO_ROOT / args.placements_v6_dir).resolve()

    if args.only_placement:
        names = [args.only_placement]
    elif args.assignment_file and (REPO_ROOT / args.assignment_file).exists():
        names = load_assignment(REPO_ROOT / args.assignment_file, args.side)
        if args.pilot_count and args.pilot_count > 0:
            names = names[: args.pilot_count]
    else:
        # Fallback: walk out_dir
        names = [
            sub.name for sub in sorted(out_dir.iterdir())
            if sub.is_dir() and not sub.name.startswith("_")
        ]

    print(f"[stage3] side={args.side} placements={len(names)} "
          f"out={out_dir} elev_sign={args.elev_sign}")

    n_ok = n_skip = n_fail = n_pending = 0
    t_start = time.time()
    for k, name in enumerate(names):
        try:
            res = score_placement(name, out_dir, v6_dir, args)
        except Exception:
            tb = traceback.format_exc()
            (out_dir / name).mkdir(parents=True, exist_ok=True)
            (out_dir / name / "score_failed.flag").write_text(tb, encoding="utf-8")
            print(f"[stage3] FAIL {name}\n{tb}", file=sys.stderr)
            n_fail += 1
            continue
        status = res["status"]
        if status == "ok":
            n_ok += 1
            print(
                f"[stage3] {k+1}/{len(names)} ok {name} "
                f"frames={res['n_frames']} t={res['elapsed']:.3f}s"
            )
        elif status == "skip":
            n_skip += 1
        elif status in ("no_bbox", "missing", "no_resolution"):
            n_pending += 1
            print(f"[stage3] {k+1}/{len(names)} pending {name} ({status})")
        else:
            n_skip += 1

    print(
        f"[stage3] done in {(time.time()-t_start)/60:.2f}min "
        f"ok={n_ok} skip={n_skip} pending={n_pending} fail={n_fail}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
