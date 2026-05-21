#!/usr/bin/env python3
"""One-time migration: flip cam_to_obj_{elevation,azimuth} convention v1 -> v2.

v1 (old): elevation/azimuth describe the obj->cam direction
          (camera ABOVE object => elevation = +90).
v2 (new): elevation/azimuth describe the cam->obj direction
          (camera ABOVE object => elevation = -90).

Transform per field, applied per row:
    score_cam_to_obj_elevation_deg : x -> -x
    score_cam_to_obj_azimuth_deg   : x -> (x + 180) % 360
    elevation_deg (raw float)       : x -> -x
    azimuth_deg (raw float)         : x -> (x + 180) % 360

Safety design:
  - dry-run by default; --execute writes
  - per-file .bak backup before overwrite (atomic rename)
  - per-directory side-car flag `_cam_to_obj_convention_v2.flag` written
    AFTER all annotations.json files in the dir migrate successfully.
    Re-running the script skips dirs with that flag (idempotent).
  - records md5 of the original file in the flag's JSON payload, so a
    later audit can detect tampering.
  - only touches the four field names listed above. Anything else (camera
    positions, bboxes, masks, scores) is left bit-identical.

Usage:
    python scripts/migrate_cam_to_obj_sign_v2.py                                  # default dry-run on default targets
    python scripts/migrate_cam_to_obj_sign_v2.py --execute                        # apply with backups
    python scripts/migrate_cam_to_obj_sign_v2.py --targets outputs/smoke_v6_*     # custom targets
    python scripts/migrate_cam_to_obj_sign_v2.py --execute --no-backup            # skip .bak (NOT recommended)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDS_INT = ("score_cam_to_obj_elevation_deg", "score_cam_to_obj_azimuth_deg")
FIELDS_FLOAT = ("elevation_deg", "azimuth_deg")
ELEV_KEYS = ("score_cam_to_obj_elevation_deg", "elevation_deg")
AZIM_KEYS = ("score_cam_to_obj_azimuth_deg", "azimuth_deg")

FLAG_NAME = "_cam_to_obj_convention_v2.flag"

DEFAULT_TARGETS = [
    REPO_ROOT / "outputs/smoke_v6_pitch_lerp",
    REPO_ROOT / "outputs/smoke_v6_local_dense",
    REPO_ROOT / "outputs/v5_smoke_3090x8",
    REPO_ROOT / "outputs/v5_3090x8_260429_092917",
]


def _flip_elev(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return -x
    if isinstance(x, float):
        return -x
    return x


def _flip_azim(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return (x + 180) % 360
    if isinstance(x, float):
        return (x + 180.0) % 360.0
    return x


def transform_row(row: dict) -> tuple[dict, dict]:
    """Returns (new_row, changes) where changes is {field: (before, after)}."""
    changes = {}
    new_row = dict(row)
    for k in ELEV_KEYS:
        if k in row and row[k] is not None:
            before = row[k]
            after = _flip_elev(before)
            if after != before:
                new_row[k] = after
                changes[k] = (before, after)
    for k in AZIM_KEYS:
        if k in row and row[k] is not None:
            before = row[k]
            after = _flip_azim(before)
            # 0 maps to 180 maps to 0; document but record
            if after != before:
                new_row[k] = after
                changes[k] = (before, after)
    return new_row, changes


def file_already_migrated(annotations_dir: Path) -> bool:
    return (annotations_dir / FLAG_NAME).exists()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_flag(annotations_dir: Path, migrated_files: dict):
    payload = {
        "convention": "cam_to_obj_v2",
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transform": {
            "elevation_deg": "x -> -x",
            "azimuth_deg": "x -> (x + 180) % 360",
            "score_cam_to_obj_elevation_deg": "x -> -x",
            "score_cam_to_obj_azimuth_deg": "x -> (x + 180) % 360",
        },
        "files": migrated_files,
    }
    (annotations_dir / FLAG_NAME).write_text(json.dumps(payload, indent=2))


def find_annotation_files(target_dir: Path) -> list[Path]:
    """Return every annotations.json under target_dir (recursive)."""
    return sorted(target_dir.rglob("annotations.json"))


def migrate_file(path: Path, execute: bool, backup: bool) -> dict:
    """Read, transform, optionally write back. Returns summary dict."""
    raw = path.read_text()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        return {"path": str(path), "skipped": "not a list", "rows": 0, "changed": 0}

    n_changed = 0
    new_rows = []
    sample_change = None
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            new_rows.append(row)
            continue
        new_row, changes = transform_row(row)
        if changes:
            n_changed += 1
            if sample_change is None:
                sample_change = {"row_index": i, "changes": changes}
        new_rows.append(new_row)

    summary = {
        "path": str(path),
        "rows": len(rows),
        "changed": n_changed,
        "sample": sample_change,
    }

    if execute and n_changed > 0:
        # Compute hash of original BEFORE writing, so the .bak is verifiably the
        # pre-migration content.
        before_hash = md5_of(path)
        if backup:
            bak_path = path.with_suffix(path.suffix + ".bak")
            # Atomic: only overwrite .bak if it doesn't exist yet (no double-backup)
            if not bak_path.exists():
                bak_path.write_text(raw)
        # Write new content atomically: tmp file + rename
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(new_rows, indent=2))
        tmp.replace(path)
        summary["before_hash"] = before_hash
        summary["after_hash"] = md5_of(path)
    return summary


def migrate_target(target_dir: Path, execute: bool, backup: bool) -> dict:
    """Walk target_dir, migrate every annotations.json under it, write flag.

    If a per-placement run_dir already has annotations.json and a flag at
    the TARGET ROOT level, the whole tree is treated as migrated.
    """
    if file_already_migrated(target_dir):
        return {"target": str(target_dir), "status": "already_migrated"}

    ann_paths = find_annotation_files(target_dir)
    if not ann_paths:
        return {"target": str(target_dir), "status": "no_annotations_found"}

    file_summaries = []
    total_rows = 0
    total_changed = 0
    for p in ann_paths:
        s = migrate_file(p, execute=execute, backup=backup)
        file_summaries.append(s)
        total_rows += s.get("rows", 0)
        total_changed += s.get("changed", 0)

    summary = {
        "target": str(target_dir),
        "status": "executed" if execute else "dry_run",
        "files": len(ann_paths),
        "rows": total_rows,
        "rows_changed": total_changed,
        "details": file_summaries,
    }

    if execute and total_changed > 0:
        files_map = {
            str(Path(s["path"]).relative_to(target_dir)): {
                "rows": s.get("rows"),
                "changed": s.get("changed"),
                "before_hash": s.get("before_hash"),
                "after_hash": s.get("after_hash"),
            }
            for s in file_summaries
        }
        write_flag(target_dir, files_map)

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Write changes to disk. Without this, runs in dry-run mode.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip writing .bak files. NOT RECOMMENDED.")
    p.add_argument("--targets", nargs="+", type=Path, default=None,
                   help="Directories to migrate (recursive). Default: smoke_v6_pitch_lerp, "
                        "smoke_v6_local_dense, v5_smoke_3090x8, v5_3090x8_260429_092917.")
    args = p.parse_args()

    targets = args.targets or DEFAULT_TARGETS
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Backups: {'no' if args.no_backup else 'yes (.bak)'}")
    print(f"Targets: {[str(t) for t in targets]}")
    print()

    grand_files = 0
    grand_rows = 0
    grand_changed = 0
    grand_skipped_dirs = 0
    for t in targets:
        if not t.exists():
            print(f"[skip] {t} (does not exist)")
            continue
        s = migrate_target(t, execute=args.execute, backup=not args.no_backup)
        if s.get("status") == "already_migrated":
            grand_skipped_dirs += 1
            print(f"[already migrated] {t}")
            continue
        print(f"[{s.get('status')}] {t}: {s.get('files', 0)} files, "
              f"{s.get('rows', 0)} rows, {s.get('rows_changed', 0)} changed")
        # Show one sample change per target
        for fs in s.get("details", []):
            if fs.get("sample"):
                sample = fs["sample"]
                print(f"    sample (row {sample['row_index']}):")
                for k, (b, a) in sample["changes"].items():
                    print(f"      {k}: {b}  ->  {a}")
                break  # only one sample per target
        grand_files += s.get("files", 0)
        grand_rows += s.get("rows", 0)
        grand_changed += s.get("rows_changed", 0)

    print()
    print(f"=== SUMMARY ===")
    print(f"Targets processed:  {len(targets) - grand_skipped_dirs}/{len(targets)} "
          f"({grand_skipped_dirs} already migrated)")
    print(f"Annotation files:   {grand_files}")
    print(f"Total rows scanned: {grand_rows}")
    print(f"Rows changed:       {grand_changed}")
    if not args.execute:
        print()
        print("DRY-RUN only. Re-run with --execute to apply.")


if __name__ == "__main__":
    sys.exit(main() or 0)
