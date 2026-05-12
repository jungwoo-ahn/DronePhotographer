#!/usr/bin/env python3
"""DronePhotographer v5.0 — render the full placement set.

Reads /data/vlm_object_placing/*.json, filters to accepted placements
(``placements[].accepted == True``), and dispatches each one to
render_object_v3.py with the recorded scene_scale, position, rotation, and
scale. Reuses helpers from render_objects_in_multiple_scenes.py.

Strictly verifies that each placement run produced exactly NUM_IMAGES PNGs
and matching annotation entries; mismatches are reported in summary.json
without polluting the merged dataset.

Usage:
    python scripts/render_v5_from_placements.py \
        --placements_dir data/vlm_object_placing \
        --assets_root . \
        --output_dir outputs \
        --run_name v5_full_run \
        --num_images_per_placement 2000 \
        --gpu_devices 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    p = argparse.ArgumentParser(description="DronePhotographer v5.0 placement-driven renderer.")
    p.add_argument("--placements_dir", required=True,
                   help="Directory with /data/vlm_object_placing/*.json files.")
    p.add_argument("--assets_root", default=".",
                   help="Base path for resolving relative scene_file/object_file (default: repo root).")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--num_images_per_placement", type=int, default=2000)
    p.add_argument("--max_placements", type=int, default=None,
                   help="Process at most N accepted placements (smoke test).")
    p.add_argument("--placement_slice", default=None,
                   help="Take every n-th entry starting at i. Format 'i:n' (0-indexed). "
                        "Used for sharding placements across parallel GPU workers.")
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
    p.add_argument("--camera_radius_range", nargs=2, type=float, default=[0.5, 6.0])
    p.add_argument("--camera_direction_offsets", nargs=3, type=float, default=[15.0, 15.0, 0.0])
    p.add_argument("--hemisphere", action="store_true")
    p.add_argument("--adaptive_sampling", action="store_true")
    p.add_argument("--adaptive_threshold", type=float, default=0.02)
    p.add_argument("--use_aabb_center", action="store_true", default=True)
    p.add_argument("--resume", action="store_true",
                   help="Skip placements with an existing valid annotations.json.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


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
            # NOTE: previously filtered out placements with empty final_images
            # (~192 of 940). Investigation 2026-04-28 showed they have
            # equivalent VLM quality (median 9.5, mean 9.23 vs 9.44) and all
            # have >=1 visible view — the empty final_images appears to be a
            # quirk of the placement pipeline, not a quality signal. Including
            # them grows training data 748 -> 940 (+25.7%).
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
    images = sorted(p for p in (run_dir / "images").glob("*.png"))
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


def main() -> int:
    args = parse_args()
    placements_dir = Path(args.placements_dir).resolve()
    if not placements_dir.is_dir():
        print(f"placements_dir not found: {placements_dir}", file=sys.stderr)
        return 2

    entries = collect_accepted_placements(placements_dir)
    if args.max_placements:
        entries = entries[: args.max_placements]

    slice_idx: int | None = None
    slice_n = 1
    if args.placement_slice:
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
    master_name = args.run_name or f"v5_{timestamp}"
    master_dir = Path(args.output_dir) / master_name
    master_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_slice_{slice_idx}_of_{slice_n}" if slice_idx is not None else ""
    manifest = {
        "version": "v5.0",
        "created_at": timestamp,
        "placements_dir": str(placements_dir),
        "num_placements": len(entries),
        "num_images_per_placement": args.num_images_per_placement,
        "gpu_devices": args.gpu_devices,
        "slice": f"{slice_idx}/{slice_n}" if slice_idx is not None else None,
    }
    (master_dir / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))
    (master_dir / f"placements_v5{suffix}.json").write_text(json.dumps(entries, indent=2))

    print(f"Rendering {len(entries)} accepted placements x {args.num_images_per_placement} images each.")
    print(f"GPUs: {args.gpu_devices}  output: {master_dir}")

    results = []
    total_start = time.time()
    for idx, entry in enumerate(entries):
        label = build_run_name(entry, idx)
        run_dir = master_dir / label

        if args.resume and (run_dir / "annotations.json").exists():
            ok, _msg = verify_run_dir(run_dir, args.num_images_per_placement)
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

        if args.dry_run:
            print("  (dry run)", " ".join(common_args))
            results.append({"placement": label, "status": "dry_run"})
            continue

        processes = launch_workers(args, common_args, run_dir)
        failed = monitor_and_wait(processes, run_dir, args.num_images_per_placement, label)
        if len(args.gpu_devices) > 1:
            merged_n = merge_worker_annotations(run_dir, len(args.gpu_devices))
            print(f"  Merged {merged_n} annotations")

        ok, why = verify_run_dir(run_dir, args.num_images_per_placement)
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
