#!/usr/bin/env python3
"""Backfill mesh-projected bboxes into already-rendered Stage 2 outputs.

For each placement under ``--out-dir`` that has ``done.flag`` but whose
``render_records[*][*]`` lack ``bbox_xyxy_full``:
  1. Open scene + place object (reusing smoke.setup_blender_scene)
  2. Build the in-frame projector against the camera at the placement's
     resolution
  3. For each stored frame in ``trajectory_32f`` (the frames we actually
     rendered, per ``render_records``), project the subject mesh and patch
     ``bbox_xyxy_full`` / ``occupancy_clipped`` / ``in_frame``
  4. Write back ``data.json``, drop ``backfilled.flag``

This script does NOT render. It only projects geometry — pure CPU/numpy work
after the initial scene load. Expected: ~5-10s per placement (setup-dominated).

Run inside Blender:

    blender/blender -b -P scripts/v7_stage2_backfill_bbox.py -- \\
        --out-dir outputs/v7_stage2_smoke_diverse \\
        --placements-v6-dir data/vlm_object_placing_v6_260428_061326 \\
        --assets-root /home/nas1/jungwooahn/projects/DronePhotographer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SMOKE = REPO_ROOT / "scripts" / "v7_sample_pairs_smoke.py"
sys.path.insert(0, str(SMOKE.parent))
import v7_sample_pairs_smoke as smoke  # noqa: E402

# Reuse the helpers (placement-dict build, v6 lookup, slice selection).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from v7_stage2_render import (  # noqa: E402
    build_placement_dict,
    load_assignment,
    load_v6_placement,
)

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True,
                   help="Stage 2 output dir to backfill (e.g. outputs/v7_stage2_smoke_diverse).")
    p.add_argument("--stage1-dir", default="outputs/v7_stage1_sample")
    p.add_argument("--placements-v6-dir",
                   default="data/vlm_object_placing_v6_260428_061326")
    p.add_argument("--assets-root", default=str(REPO_ROOT))
    p.add_argument("--assignment-file", default=None,
                   help="Optional manifest; if given with --side, restrict to that side.")
    p.add_argument("--side", default=None)
    p.add_argument("--only-placement", default=None,
                   help="Backfill exactly one placement directory by name.")
    p.add_argument("--force", action="store_true",
                   help="Re-backfill even if bbox already present / backfilled.flag exists.")
    return p.parse_args(argv)


def collect_targets(out_dir: Path, assignment_path: Path | None,
                    side: str | None, only: str | None) -> list[str]:
    """Return placement directory names eligible for backfill."""
    if only:
        return [only]
    candidates: list[str] = []
    if assignment_path is not None and side:
        names = load_assignment(assignment_path, side)
        for n in names:
            if (out_dir / n).is_dir():
                candidates.append(n)
    else:
        for sub in sorted(out_dir.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            candidates.append(sub.name)
    return candidates


def needs_backfill(placement_dir: Path, force: bool) -> bool:
    if force:
        return True
    if not (placement_dir / "done.flag").exists():
        return False
    if (placement_dir / "backfilled.flag").exists():
        return False
    data_path = placement_dir / "data.json"
    if not data_path.exists():
        return False
    try:
        d = json.loads(data_path.read_text())
    except Exception:
        return False
    for pair_recs in (d.get("render_records") or []):
        for rec in pair_recs:
            if "bbox_xyxy_full" in rec:
                return False  # already backfilled
            return True       # first record missing → backfill
    return False


def backfill_placement(
    name: str,
    out_dir: Path,
    v6_dir: Path,
    assets_root: Path,
) -> dict:
    placement_dir = out_dir / name
    data_path = placement_dir / "data.json"
    data = json.loads(data_path.read_text())

    W = int(data.get("render_width") or 0)
    H = int(data.get("render_height") or 0)
    if W <= 0 or H <= 0:
        raise ValueError(f"missing render_width/height in {data_path}")

    v6 = load_v6_placement(
        v6_dir / f"{name}.json",
        int(data.get("placement_idx", 0)),
    )
    placement = build_placement_dict(data, v6)

    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    meta = smoke.setup_blender_scene(placement, assets_root)
    t_setup = time.time() - t0

    # Minimal camera setup (no render config, no GPU) — we just need view_frame
    # to derive frame_bounds for the perspective projection.
    cam_data = bpy.data.cameras.new("V7BackfillCam")
    cam_data.lens = 24.0
    cam_data.sensor_width = 12.8
    cam_data.sensor_height = 9.6
    cam_obj = bpy.data.objects.new("V7BackfillCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = W
    bpy.context.scene.render.resolution_y = H

    frame_bounds = smoke._get_camera_frame_bounds(bpy.context.scene, cam_obj)
    in_frame_check = smoke.make_in_frame_check(
        meta["subject_verts_world"], frame_bounds, (W, H),
    )

    accepted_pairs = data.get("accepted_pairs") or []
    render_records = data.get("render_records") or []
    n_patched = 0
    t_proj_total = 0.0
    for pair_idx, pair_recs in enumerate(render_records):
        if pair_idx >= len(accepted_pairs):
            continue
        traj = accepted_pairs[pair_idx].get("trajectory_32f") or []
        for rec in pair_recs:
            fi = int(rec.get("frame_idx", -1))
            if fi < 0 or fi >= len(traj):
                continue
            f = traj[fi]
            t0 = time.time()
            in_frame, occ, bbox = in_frame_check(
                np.asarray(f["pos"], dtype=np.float64),
                np.asarray(f["forward"], dtype=np.float64),
                np.asarray(f["up"], dtype=np.float64),
            )
            t_proj_total += time.time() - t0
            rec["bbox_xyxy_full"] = (
                [float(v) for v in bbox] if bbox is not None else None
            )
            rec["occupancy_clipped"] = float(occ)
            rec["in_frame"] = bool(in_frame)
            n_patched += 1

    data["render_records"] = render_records
    data["stage2_backfill_setup_s"] = float(t_setup)
    data["stage2_backfill_proj_s"] = float(t_proj_total)
    data["stage2_backfill_n_frames"] = int(n_patched)
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (placement_dir / "backfilled.flag").write_text(
        f"frames={n_patched}  setup={t_setup:.2f}s  proj={t_proj_total:.2f}s\n",
        encoding="utf-8",
    )
    return {"name": name, "status": "ok", "n_frames": n_patched,
            "t_setup": t_setup, "t_proj": t_proj_total}


def main() -> int:
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    v6_dir = (REPO_ROOT / args.placements_v6_dir).resolve()
    assets_root = Path(args.assets_root).resolve()
    assignment_path = (
        (REPO_ROOT / args.assignment_file).resolve() if args.assignment_file else None
    )

    names = collect_targets(out_dir, assignment_path, args.side, args.only_placement)
    print(f"[backfill] candidates={len(names)} out_dir={out_dir}")
    if not names:
        print("[backfill] nothing to do")
        return 0

    n_ok = n_skip = n_fail = 0
    t_start = time.time()
    for k, name in enumerate(names):
        placement_dir = out_dir / name
        if not needs_backfill(placement_dir, args.force):
            n_skip += 1
            print(f"[backfill] {k+1}/{len(names)} skip {name}")
            continue
        try:
            res = backfill_placement(name, out_dir, v6_dir, assets_root)
        except Exception:
            tb = traceback.format_exc()
            (placement_dir / "backfill_failed.flag").write_text(tb, encoding="utf-8")
            print(f"[backfill] FAIL {name}\n{tb}", file=sys.stderr)
            n_fail += 1
            continue
        n_ok += 1
        print(
            f"[backfill] {k+1}/{len(names)} ok {name} "
            f"frames={res['n_frames']} setup={res['t_setup']:.2f}s "
            f"proj={res['t_proj']:.3f}s"
        )

    print(
        f"[backfill] done in {(time.time()-t_start)/60:.2f}min "
        f"ok={n_ok} skip={n_skip} fail={n_fail}"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
