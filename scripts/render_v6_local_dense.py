#!/usr/bin/env python3
"""DronePhotographer v6.0 — placement-anchored dense rendering.

Reads data/vlm_object_placing_v6_*/*.json placements, filters to accepted ones
(``placements[].accepted == True`` with a position), and dispatches each to
render_object_v3.py running in --local_dense mode:

  1. discover N valid anchor camera positions on the hemisphere around the
     placed object (random sample → is_camera_valid filter).
  2. for each anchor, render M poses densely inside a 3 m ball around it
     (look-at object + yaw/pitch/roll jitter).

Total per placement: N × M images (default 4 × 100 = 400).

Strictly verifies that each placement run produced exactly that many images
with matching annotation entries; mismatches are reported in summary.json
without polluting the merged dataset.

Usage:
    python scripts/render_v6_local_dense.py \
        --placements_dir data/vlm_object_placing_v6_260428_061326 \
        --assets_root . \
        --output_dir outputs \
        --run_name v6_local_dense_full \
        --num_anchors_per_placement 4 \
        --num_images_per_anchor 100 \
        --gpu_devices 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from render_objects_in_multiple_scenes import (
    build_common_args,
    build_run_name,
    launch_workers,
    merge_worker_annotations,
    monitor_and_wait,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DronePhotographer v6.0 local-dense renderer.")
    p.add_argument("--placements_dir", default="data/vlm_object_placing_v6_260428_061326",
                   help="Directory with v6 placement JSON files.")
    p.add_argument("--assets_root", default=".",
                   help="Base path for resolving relative scene_file/object_file (default: repo root).")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--run_name", default=None)

    # v6 local-dense knobs (per the approved plan)
    p.add_argument("--num_anchors_per_placement", type=int, default=4)
    p.add_argument("--num_images_per_anchor", type=int, default=100)
    p.add_argument("--anchor_radius_range", nargs=2, type=float, default=[1.0, 8.0])
    p.add_argument("--anchor_ball_radius", type=float, default=3.0)
    p.add_argument("--anchor_max_attempts", type=int, default=500)
    p.add_argument("--anchor_min_clearance", type=float, default=0.8)
    p.add_argument("--camera_direction_offsets", nargs=3, type=float, default=[15.0, 15.0, 0.0],
                   help="yaw pitch roll jitter (deg) applied to look-at inside the ball")
    # Distance-dependent camera pitch (overrides the 3rd value of camera_direction_offsets
    # in local_dense mode). Format: (r, low_deg, high_deg) at two distances; linear lerp.
    p.add_argument("--pitch_lerp_near", nargs=3, type=float, default=[1.0, -15.0, 45.0],
                   metavar=("R_NEAR", "LOW_NEAR", "HIGH_NEAR"),
                   help="(r, low, high) at the near distance (default 1m -15°..+45°)")
    p.add_argument("--pitch_lerp_far", nargs=3, type=float, default=[8.0, -15.0, 15.0],
                   metavar=("R_FAR", "LOW_FAR", "HIGH_FAR"),
                   help="(r, low, high) at the far distance (default 8m -15°..+15°)")

    p.add_argument("--max_placements", type=int, default=None,
                   help="Process at most N accepted placements (smoke test).")
    p.add_argument("--placement_slice", default=None,
                   help="Take every n-th entry starting at i. Format 'i:n' (0-indexed).")
    p.add_argument("--placement_ids", nargs="+", type=int, default=None,
                   help="Only render entries whose `id` is in this list; overrides max_placements/slice.")
    p.add_argument("--gpu_devices", nargs="+", type=int, required=True)
    p.add_argument("--gpu_backend", default="OPTIX")
    p.add_argument("--blender_bin", default="blender/blender")
    p.add_argument("--blender_threads", type=int, default=4)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--max_bounces", type=int, default=4)
    p.add_argument("--diffuse_bounces", type=int, default=2)
    p.add_argument("--glossy_bounces", type=int, default=2)
    p.add_argument("--transmission_bounces", type=int, default=2)
    p.add_argument("--persistent_data", action="store_true", default=True)
    p.add_argument("--sky_strength", type=float, default=0.3)
    # Unused by local_dense mode but required by build_common_args:
    p.add_argument("--camera_radius_range", nargs=2, type=float, default=[1.0, 8.0],
                   help="(ignored in local_dense mode; kept for build_common_args compat)")
    p.add_argument("--hemisphere", action="store_true", default=True,
                   help="(forced True in local_dense anchor discovery)")
    p.add_argument("--adaptive_sampling", action="store_true")
    p.add_argument("--adaptive_threshold", type=float, default=0.02)
    p.add_argument("--use_aabb_center", action="store_true", default=True)
    p.add_argument("--resume", action="store_true",
                   help="Skip placements with an existing valid annotations.json.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    # Derived: total images per placement. build_common_args uses this as --num_images.
    args.num_images_per_placement = args.num_anchors_per_placement * args.num_images_per_anchor
    return args


def collect_accepted_placements(placements_dir: Path) -> list[dict]:
    """Walk placement JSONs and emit one render-entry per accepted placement.

    Output entries match the schema build_common_args expects, plus
    ``scene_scale`` and ``rotation_xyz_rad``.
    """
    entries: list[dict] = []
    seen_ids = 0
    json_files = sorted(placements_dir.glob("*.json"))
    for jf in json_files:
        if jf.name in {"summary.json", "manifest.json"}:
            continue
        try:
            doc = json.loads(jf.read_text())
        except Exception as exc:
            print(f"[skip] failed to read {jf.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            continue
        scene_file = doc.get("scene_file")
        object_file = doc.get("object_file")
        scene_scale = doc.get("scene_scale", 1.0)
        scene_label = doc.get("scene", jf.stem.split("__")[0])
        object_label = doc.get("object", jf.stem.split("__")[-1])
        for placement in doc.get("placements", []) or []:
            if not placement.get("accepted"):
                continue
            position = placement.get("position")
            rotation = placement.get("rotation", [0.0, 0.0, 0.0])
            scale = placement.get("scale", 1.0)
            if scene_file is None or object_file is None or position is None:
                continue
            entries.append({
                "id": seen_ids,
                "source_json": jf.name,
                "scene": scene_label,
                "object": object_label,
                "scene_path": scene_file,
                "object_path": object_file,
                "scene_scale": float(scene_scale),
                "position": [float(v) for v in position],
                "rotation_xyz_rad": [float(v) for v in rotation],
                "scale": float(scale),
                "candidate_idx": placement.get("candidate_idx"),
            })
            seen_ids += 1
    return entries


def verify_run_dir(run_dir: Path, expected_images: int) -> tuple[bool, str]:
    images_dir = run_dir / "images"
    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ) if images_dir.is_dir() else []
    n_images = len(images)
    ann_path = run_dir / "annotations.json"
    if n_images != expected_images:
        return False, f"image count {n_images} != {expected_images}"
    if not ann_path.exists():
        return False, "annotations.json missing"
    try:
        ann = json.loads(ann_path.read_text())
    except Exception as exc:
        return False, f"annotations.json unreadable: {exc}"
    if not isinstance(ann, list) or len(ann) != expected_images:
        return False, f"annotation count {len(ann) if isinstance(ann, list) else 'N/A'} != {expected_images}"
    return True, "ok"


def build_local_dense_args(args) -> list[str]:
    """Extra args to append to build_common_args output to enable local_dense mode."""
    return [
        "--local_dense",
        "--num_anchors_per_placement", str(args.num_anchors_per_placement),
        "--num_images_per_anchor", str(args.num_images_per_anchor),
        "--anchor_radius_range", str(args.anchor_radius_range[0]), str(args.anchor_radius_range[1]),
        "--anchor_ball_radius", str(args.anchor_ball_radius),
        "--anchor_max_attempts", str(args.anchor_max_attempts),
        "--anchor_min_clearance", str(args.anchor_min_clearance),
        "--pitch_lerp_near", str(args.pitch_lerp_near[0]), str(args.pitch_lerp_near[1]), str(args.pitch_lerp_near[2]),
        "--pitch_lerp_far", str(args.pitch_lerp_far[0]), str(args.pitch_lerp_far[1]), str(args.pitch_lerp_far[2]),
    ]


def main() -> int:
    args = parse_args()
    placements_dir = Path(args.placements_dir).resolve()
    if not placements_dir.is_dir():
        print(f"placements_dir not found: {placements_dir}", file=sys.stderr)
        return 2

    entries = collect_accepted_placements(placements_dir)
    if args.placement_ids:
        wanted = set(args.placement_ids)
        entries = [e for e in entries if e["id"] in wanted]
        if len(entries) < len(wanted):
            missing = sorted(wanted - {e["id"] for e in entries})
            print(f"warning: placement ids not found: {missing}", file=sys.stderr)
    elif args.max_placements:
        entries = entries[: args.max_placements]

    slice_idx: int | None = None
    slice_n = 1
    if args.placement_slice and not args.placement_ids:
        try:
            s_i, s_n = args.placement_slice.split(":")
            slice_idx, slice_n = int(s_i), int(s_n)
            assert slice_n >= 1 and 0 <= slice_idx < slice_n
        except Exception:
            print(f"--placement_slice must be 'i:n' with 0 <= i < n: got {args.placement_slice!r}",
                  file=sys.stderr)
            return 2
        entries = [e for k, e in enumerate(entries) if k % slice_n == slice_idx]

    if not entries:
        if slice_idx is not None:
            print(f"Slice {slice_idx}/{slice_n}: no placements assigned, exiting cleanly.")
            return 0
        print("No accepted placements found.", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    master_name = args.run_name or f"v6_local_dense_{timestamp}"
    master_dir = Path(args.output_dir) / master_name
    master_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_slice_{slice_idx}_of_{slice_n}" if slice_idx is not None else ""
    images_per_placement = args.num_anchors_per_placement * args.num_images_per_anchor
    manifest = {
        "version": "v6.0_local_dense",
        "created_at": timestamp,
        "placements_dir": str(placements_dir),
        "num_placements": len(entries),
        "num_anchors_per_placement": args.num_anchors_per_placement,
        "num_images_per_anchor": args.num_images_per_anchor,
        "images_per_placement": images_per_placement,
        "anchor_radius_range": list(args.anchor_radius_range),
        "anchor_ball_radius": args.anchor_ball_radius,
        "anchor_min_clearance": args.anchor_min_clearance,
        "anchor_max_attempts": args.anchor_max_attempts,
        "camera_direction_offsets": list(args.camera_direction_offsets),
        "pitch_lerp_near": list(args.pitch_lerp_near),
        "pitch_lerp_far": list(args.pitch_lerp_far),
        "gpu_devices": args.gpu_devices,
        "slice": f"{slice_idx}/{slice_n}" if slice_idx is not None else None,
    }
    (master_dir / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))
    (master_dir / f"placements_v6{suffix}.json").write_text(json.dumps(entries, indent=2))

    print(f"Rendering {len(entries)} placements x "
          f"{args.num_anchors_per_placement} anchors x "
          f"{args.num_images_per_anchor} images = "
          f"{images_per_placement} images/placement.")
    print(f"GPUs: {args.gpu_devices}  output: {master_dir}")

    results = []
    total_start = time.time()
    extra_args = build_local_dense_args(args)
    for idx, entry in enumerate(entries):
        label = build_run_name(entry, idx)
        run_dir = master_dir / label

        if args.resume and (run_dir / "annotations.json").exists():
            ok, _msg = verify_run_dir(run_dir, images_per_placement)
            if ok:
                print(f"[{idx + 1}/{len(entries)}] SKIP {label} (already valid)")
                results.append({"placement": label, "status": "skipped"})
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)
        (run_dir / "placement.json").write_text(json.dumps(entry, indent=2))

        print(f"[{idx + 1}/{len(entries)}] {label}")
        try:
            common_args = build_common_args(entry, args, run_dir)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            results.append({"placement": label, "status": "error", "message": str(exc)})
            continue
        common_args = common_args + extra_args

        if args.dry_run:
            print("  (dry run)", " ".join(common_args))
            results.append({"placement": label, "status": "dry_run"})
            continue

        processes = launch_workers(args, common_args, run_dir)
        failed = monitor_and_wait(processes, run_dir, images_per_placement, label)
        if len(args.gpu_devices) > 1:
            merged_n = merge_worker_annotations(run_dir, len(args.gpu_devices))
            print(f"  Merged {merged_n} annotations")

        ok, why = verify_run_dir(run_dir, images_per_placement)
        if failed and not ok:
            results.append({
                "placement": label,
                "status": "failed",
                "reason": why,
                "failed_gpus": [g for g, _ in failed],
            })
        elif not ok:
            results.append({"placement": label, "status": "failed", "reason": why})
        else:
            results.append({"placement": label, "status": "success"})

    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)
    summary = {
        "total_time": f"{mins}m {secs}s",
        "num_placements": len(entries),
        "results": results,
    }
    (master_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    n_ok = sum(1 for r in results if r["status"] == "success")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] in {"failed", "error"})
    print(f"\nDone. success={n_ok} skipped={n_skip} failed={n_fail} [{mins}m {secs}s]")
    print(f"Output: {master_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
