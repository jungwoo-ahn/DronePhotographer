#!/usr/bin/env python3
"""Filter placement JSONs that are viable for Stage 1 sampling.

Drops:
  - JSONs with zero `placements[]` entries
  - JSONs referencing a `scene_file` that doesn't exist on disk

Writes one line per surviving placement (absolute path) to the output file.
Skipped files are summarized in stderr but not enumerated by default.

Usage:
    python scripts/v7_stage1_prescan.py \\
        data/vlm_object_placing_v6_260428_061326 \\
        outputs/v7_stage1_sample/valid_placements.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def prescan(placements_dir: Path, out_path: Path, repo_root: Path) -> dict:
    candidates = sorted(p for p in placements_dir.glob("*.json")
                        if not p.name.startswith("_"))
    n_total = len(candidates)
    valid: list[Path] = []
    skipped_empty: list[str] = []
    skipped_no_scene: list[str] = []
    parse_errors: list[str] = []
    missing_scenes: set[str] = set()

    for p in candidates:
        try:
            doc = json.loads(p.read_text())
        except Exception as exc:
            parse_errors.append(f"{p.name}: {exc}")
            continue
        if not doc.get("placements"):
            skipped_empty.append(p.name)
            continue
        scene_rel = doc.get("scene_file", "")
        scene_full = (repo_root / scene_rel) if scene_rel else None
        if not scene_full or not scene_full.exists():
            skipped_no_scene.append(p.name)
            if scene_rel:
                missing_scenes.add(scene_rel)
            continue
        valid.append(p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p.resolve()) for p in valid) + "\n",
                        encoding="utf-8")

    return {
        "n_total": n_total,
        "n_valid": len(valid),
        "n_empty": len(skipped_empty),
        "n_no_scene": len(skipped_no_scene),
        "n_parse_error": len(parse_errors),
        "missing_scenes": sorted(missing_scenes),
        "parse_errors": parse_errors[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("placements_dir",
                    help="Directory containing placement JSONs.")
    ap.add_argument("out_path",
                    help="Output text file (one absolute path per line).")
    ap.add_argument("--repo-root", default=None,
                    help="Repo root for resolving relative scene_file paths "
                         "(default: parent of this script's parent).")
    args = ap.parse_args()

    placements_dir = Path(args.placements_dir).resolve()
    out_path = Path(args.out_path).resolve()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[1]
    )

    if not placements_dir.is_dir():
        print(f"error: {placements_dir} is not a directory", file=sys.stderr)
        return 1

    stats = prescan(placements_dir, out_path, repo_root)

    print(f"placements scanned : {stats['n_total']}")
    print(f"valid              : {stats['n_valid']}")
    print(f"  ↳ written to     : {out_path}")
    print(f"skipped (empty)    : {stats['n_empty']}")
    print(f"skipped (no scene) : {stats['n_no_scene']}")
    if stats['parse_errors']:
        print(f"parse errors       : {stats['n_parse_error']}")
        for err in stats['parse_errors']:
            print(f"  - {err}")
    if stats['missing_scenes']:
        print(f"missing scene files: {len(stats['missing_scenes'])}")
        for s in stats['missing_scenes'][:5]:
            print(f"  - {s}")
        if len(stats['missing_scenes']) > 5:
            print(f"  ... ({len(stats['missing_scenes']) - 5} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
