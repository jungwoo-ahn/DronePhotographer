#!/usr/bin/env python3
"""Render multiple scene+object placements using render_object_v3.py.

Reads a placement JSON (array of entries with scene_path, object_path, position, etc.),
then for each entry launches multi-GPU render_object_v3.py workers sequentially.

Usage:
    python render_objects_in_multiple_scenes.py \
        --placements placements.json \
        --assets_root /path/to/assets \
        --output_dir outputs \
        --num_images_per_placement 1000 \
        --gpu_devices 1 2 3 4 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Batch render multiple scene+object placements.")

    # Input
    p.add_argument("--placements", required=True, help="JSON file with placement entries")
    p.add_argument("--assets_root", required=True, help="Base path for resolving relative scene_path/object_path")

    # Output
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--run_name", default=None, help="master output directory name (default: auto-timestamped)")

    # Per-placement render params
    p.add_argument("--num_images_per_placement", type=int, default=1000)
    p.add_argument("--camera_radius_range", nargs=2, type=float, default=[0.5, 6.0])
    p.add_argument("--hemisphere", action="store_true")
    p.add_argument("--camera_direction_offsets", nargs=3, type=float, default=[15.0, 15.0, 0.0])
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--adaptive_sampling", action="store_true")
    p.add_argument("--adaptive_threshold", type=float, default=0.02)
    p.add_argument("--max_bounces", type=int, default=2)
    p.add_argument("--diffuse_bounces", type=int, default=1)
    p.add_argument("--glossy_bounces", type=int, default=1)
    p.add_argument("--transmission_bounces", type=int, default=1)
    p.add_argument("--persistent_data", action="store_true", default=True)
    p.add_argument("--sky_strength", type=float, default=0.1)

    # GPU / system
    p.add_argument("--gpu_devices", nargs="+", type=int, required=True)
    p.add_argument("--gpu_backend", default="OPTIX")
    p.add_argument("--blender_bin", default="blender/blender")
    p.add_argument("--blender_threads", type=int, default=4)
    p.add_argument("--use_aabb_center", action="store_true", help="use AABB center as camera orbit center")

    # Control
    p.add_argument("--dry_run", action="store_true", help="print commands without executing")
    p.add_argument("--resume", action="store_true", help="skip placements whose annotations.json already exists")
    p.add_argument("--placement_ids", nargs="+", type=int, help="only render specific placement IDs")

    return p.parse_args()


def resolve_path(relative_path, assets_root):
    """Resolve a path against assets_root if relative."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (Path(assets_root) / p).resolve()


def build_run_name(entry, idx):
    """Generate a directory name for a placement entry."""
    entry_id = entry.get("id", idx)
    scene = entry.get("scene") or entry.get("name") or Path(entry.get("scene_path") or entry["blend"]).stem
    # Truncate long scene names
    scene = scene[:30]
    if "object_path" in entry:
        obj = entry.get("object") or Path(entry["object_path"]).stem
        obj = obj[:30]
        return f"p{entry_id}_{scene}_{obj}"
    elif "object_name" in entry:
        return f"p{entry_id}_{scene}_{entry['object_name']}"
    return f"p{entry_id}_{scene}"


def build_common_args(entry, args, run_dir):
    """Build the CLI args for render_object_v3.py from a placement entry."""
    scene_path = resolve_path(entry.get("scene_path") or entry["blend"], args.assets_root)

    cmd_args = [
        "--input_scene", str(scene_path),
        "--output_run_dir", str(run_dir),
        "--num_images", str(args.num_images_per_placement),
        "--gpu_backend", args.gpu_backend,
        "--camera_radius_range", str(args.camera_radius_range[0]), str(args.camera_radius_range[1]),
        "--camera_direction_offsets",
        str(args.camera_direction_offsets[0]),
        str(args.camera_direction_offsets[1]),
        str(args.camera_direction_offsets[2]),
        "--samples", str(args.samples),
        "--max_bounces", str(args.max_bounces),
        "--diffuse_bounces", str(args.diffuse_bounces),
        "--glossy_bounces", str(args.glossy_bounces),
        "--transmission_bounces", str(args.transmission_bounces),
        "--sky_strength", str(args.sky_strength),
        "--blender_threads", str(args.blender_threads),
        "--num_workers", str(len(args.gpu_devices)),
    ]

    if args.hemisphere:
        cmd_args.append("--hemisphere")
    if args.adaptive_sampling:
        cmd_args.extend(["--adaptive_sampling", "--adaptive_threshold", str(args.adaptive_threshold)])
    if args.persistent_data:
        cmd_args.append("--persistent_data")
    if args.use_aabb_center:
        cmd_args.append("--use_aabb_center")

    # Placement-specific: import mode vs object_name mode
    if "object_path" in entry and "position" in entry:
        obj_path = resolve_path(entry["object_path"], args.assets_root)
        pos = entry["position"]
        cmd_args.extend(["--input_object", str(obj_path)])
        cmd_args.extend(["--object_position", str(pos[0]), str(pos[1]), str(pos[2])])
        if "rotation_z_deg" in entry:
            cmd_args.extend(["--rotation_z_deg", str(entry["rotation_z_deg"])])
        if "scale" in entry:
            cmd_args.extend(["--scale", str(entry["scale"])])
    elif "object_name" in entry:
        cmd_args.extend(["--object_name", entry["object_name"]])
    else:
        raise ValueError(f"Placement entry must have (object_path + position) or object_name: {entry}")

    return cmd_args


def launch_workers(args, common_args, run_dir):
    """Launch one Blender process per GPU. Returns list of (Popen, gpu_id, log_file)."""
    repo_root = Path(__file__).resolve().parent
    renderer = repo_root / "render_object_v3.py"

    log_dir = run_dir / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for i, gpu_id in enumerate(args.gpu_devices):
        log_path = log_dir / f"worker_{i}.log"
        cmd = [
            str(Path(args.blender_bin).resolve() if not Path(args.blender_bin).is_absolute()
                else args.blender_bin),
            "-b", "-t", str(args.blender_threads),
            "-P", str(renderer), "--",
            *common_args,
            "--worker_index", str(i),
            "--gpu_devices", "0",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = str(args.blender_threads)
        env["OPENBLAS_NUM_THREADS"] = str(args.blender_threads)
        env["MKL_NUM_THREADS"] = str(args.blender_threads)

        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(repo_root))
        processes.append((proc, gpu_id, log_path, log_f))

    return processes


def monitor_and_wait(processes, run_dir, num_images, label):
    """Monitor rendering progress and wait for all workers to finish."""
    images_dir = run_dir / "images"
    start = time.time()

    while True:
        still_running = any(p.poll() is None for p, _, _, _ in processes)
        if not still_running:
            break

        done = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        print(f"\r  [{label}] {done}/{num_images} images [{mins}m {secs}s]", end="", flush=True)
        time.sleep(2)

    # Final count
    done = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\r  [{label}] {done}/{num_images} images done. [{mins}m {secs}s]")

    # Check for failures
    failed = []
    for proc, gpu_id, log_path, log_f in processes:
        log_f.close()
        if proc.returncode != 0:
            failed.append((gpu_id, log_path))

    return failed


def merge_worker_annotations(run_dir, num_workers):
    """Merge per-worker annotation files into one annotations.json."""
    all_annotations = []
    for w in range(num_workers):
        p = run_dir / f"annotations_worker{w}.json"
        if p.exists():
            all_annotations.extend(json.loads(p.read_text()))

    all_annotations.sort(key=lambda x: x.get("image", ""))

    with (run_dir / "annotations.json").open("w") as f:
        json.dump(all_annotations, f, indent=2)

    for w in range(num_workers):
        p = run_dir / f"annotations_worker{w}.json"
        if p.exists():
            p.unlink()

    return len(all_annotations)


def main():
    args = parse_args()

    # Load placements
    placements = json.loads(Path(args.placements).read_text())
    if not isinstance(placements, list):
        placements = [placements]

    # Filter by IDs if specified
    if args.placement_ids:
        id_set = set(args.placement_ids)
        placements = [p for p in placements if p.get("id") in id_set]
        if not placements:
            print(f"No placements matched IDs {args.placement_ids}")
            sys.exit(1)

    gpu_ids = args.gpu_devices
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_output = Path(args.output_dir)
    master_name = args.run_name or f"multi_{timestamp}"
    master_dir = base_output / master_name
    master_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "created_at": timestamp,
        "num_placements": len(placements),
        "num_images_per_placement": args.num_images_per_placement,
        "gpu_devices": gpu_ids,
        "placements": placements,
    }
    with (master_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Rendering {len(placements)} placements, {args.num_images_per_placement} images each")
    print(f"GPUs: {gpu_ids} ({len(gpu_ids)} workers per placement)")
    print(f"Output: {master_dir}")
    print()

    results = []
    total_start = time.time()

    for idx, entry in enumerate(placements):
        label = build_run_name(entry, idx)
        run_dir = master_dir / label

        # Resume: skip if already done
        if args.resume and (run_dir / "annotations.json").exists():
            print(f"[{idx+1}/{len(placements)}] SKIP {label} (already exists)")
            results.append({"placement": label, "status": "skipped"})
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)

        print(f"[{idx+1}/{len(placements)}] {label}")

        try:
            common_args = build_common_args(entry, args, run_dir)
        except ValueError as e:
            print(f"  ERROR: {e}")
            results.append({"placement": label, "status": "error", "message": str(e)})
            continue

        # Dry run: print command and continue
        if args.dry_run:
            repo_root = Path(__file__).resolve().parent
            sample_cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_ids[0]} "
                f"{args.blender_bin} -b -t {args.blender_threads} "
                f"-P {repo_root / 'render_object_v3.py'} -- "
                + " ".join(common_args)
                + f" --worker_index 0 --gpu_devices 0"
            )
            print(f"  (dry run) {sample_cmd}")
            results.append({"placement": label, "status": "dry_run"})
            continue

        # Save placement entry for reference
        with (run_dir / "placement.json").open("w") as f:
            json.dump(entry, f, indent=2)

        # Launch workers
        processes = launch_workers(args, common_args, run_dir)
        failed = monitor_and_wait(processes, run_dir, args.num_images_per_placement, label)

        if failed:
            print(f"  WARNING: {len(failed)} worker(s) failed:")
            for gpu_id, log_path in failed:
                print(f"    GPU {gpu_id}: {log_path}")
            results.append({"placement": label, "status": "partial_failure", "failed_gpus": [g for g, _ in failed]})
        else:
            results.append({"placement": label, "status": "success"})

        # Merge annotations
        if len(gpu_ids) > 1:
            n = merge_worker_annotations(run_dir, len(gpu_ids))
            print(f"  Merged {n} annotations")

    # Save summary
    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)
    summary = {
        "total_time": f"{mins}m {secs}s",
        "results": results,
    }
    with (master_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone. {success}/{len(placements)} success, {skipped} skipped. [{mins}m {secs}s]")
    print(f"Output: {master_dir}")


if __name__ == "__main__":
    main()

"""
python render_objects_in_multiple_scenes.py \
    --placements placements.json \
    --assets_root /home/nas5/jungwooahn/datasets/DronePhotos/assets \
    --output_dir outputs \
    --num_images_per_placement 1000 \
    --gpu_devices 1 2 3 4 5 \
    --hemisphere \
    --adaptive_sampling \
    --dry_run
"""
