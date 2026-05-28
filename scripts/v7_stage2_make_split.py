#!/usr/bin/env python3
"""Generate the v7 Stage 2 placement-assignment manifest.

Walks ``<stage1-dir>/*/data.json``, sorts placements by
``(scene_file, placement_name)`` so each side's slice has scene-locality,
then halves into two disjoint sides for parallel rendering on separate
machines.

Output schema (committed to git so both sides agree):

    {
      "version": 1,
      "total_placements": <N>,
      "split_method": "sorted_by_(scene_file,placement_name)_then_halve",
      "generated_at": "<iso>",
      "sides": {
        "jungwooahn": {"count": <N/2>, "placements": [...]},
        "jooyeol":    {"count": <N/2>, "placements": [...]}
      }
    }

Usage:
    python scripts/v7_stage2_make_split.py \\
        --stage1-dir outputs/v7_stage1_sample \\
        --out splits/v7_stage2_assignments.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect(stage1_dir: Path) -> list[tuple[str, str]]:
    """Return [(scene_file, placement_name), ...] for every Stage 1 placement."""
    out: list[tuple[str, str]] = []
    for sub in sorted(stage1_dir.iterdir()):
        if not sub.is_dir():
            continue
        dj = sub / "data.json"
        if not dj.exists():
            continue
        try:
            d = json.loads(dj.read_text())
        except Exception as exc:
            print(f"[split] skip {sub.name}: {exc}", file=sys.stderr)
            continue
        scene_file = str(d.get("scene_file", ""))
        placement = str(d.get("placement", sub.name))
        out.append((scene_file, placement))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-dir", default="outputs/v7_stage1_sample")
    ap.add_argument("--out", default="splits/v7_stage2_assignments.json")
    ap.add_argument(
        "--sides",
        nargs=2,
        default=["jungwooahn", "jooyeol"],
        metavar=("LOWER", "UPPER"),
        help="Names for the two halves (default: jungwooahn jooyeol)",
    )
    args = ap.parse_args()

    stage1_dir = (REPO_ROOT / args.stage1_dir).resolve()
    out_path = (REPO_ROOT / args.out).resolve()

    rows = collect(stage1_dir)
    rows.sort(key=lambda r: (r[0], r[1]))
    placements = [name for _, name in rows]

    n = len(placements)
    mid = n // 2  # lower side gets the floor half if n is odd
    lower_name, upper_name = args.sides
    lower = placements[:mid]
    upper = placements[mid:]

    manifest = {
        "version": 1,
        "total_placements": n,
        "split_method": "sorted_by_(scene_file,placement_name)_then_halve",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stage1_dir": str(stage1_dir.relative_to(REPO_ROOT)),
        "sides": {
            lower_name: {"count": len(lower), "placements": lower},
            upper_name: {"count": len(upper), "placements": upper},
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[split] wrote {out_path}  "
        f"total={n}  {lower_name}={len(lower)}  {upper_name}={len(upper)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
