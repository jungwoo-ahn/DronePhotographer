"""Apply mesh-tight bbox to annotations.json.

For each placement that has a mesh_tight_bbox.json sidecar:
  - replace `bbox_2d_full_projected` with the mesh-tight bbox
  - recompute the 6 bbox-derived v5 scores via compute_v5_scores
  - leave `cam_to_obj_azimuth_deg` / `cam_to_obj_elevation_deg` unchanged
    (those come from `azimuth_deg` / `elevation_deg`, not from the bbox).

Writes back to annotations.json in-place. Original bbox is preserved under
`bbox_2d_full_projected_aabb_backup` so we can diff or revert later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scoring.bbox_control import compute_v5_scores


def process_placement(placement_dir: Path) -> dict:
    ann_path = placement_dir / "annotations.json"
    mt_path = placement_dir / "mesh_tight_bbox.json"
    if not ann_path.exists() or not mt_path.exists():
        return {"skipped": True, "reason": "missing json"}

    ann = json.loads(ann_path.read_text())
    mt = json.loads(mt_path.read_text())
    mt_by_img = {m["image"]: m["bbox_2d_full_mesh_tight"] for m in mt}

    W, H = 1024, 768
    updated = 0
    for entry in ann:
        bbox_new = mt_by_img.get(entry["image"])
        if bbox_new is None:
            continue
        # Backup old AABB once
        if "bbox_2d_full_projected_aabb_backup" not in entry:
            entry["bbox_2d_full_projected_aabb_backup"] = entry.get("bbox_2d_full_projected")
        entry["bbox_2d_full_projected"] = [float(v) for v in bbox_new]

        # Recompute v5 scores using new bbox + existing azimuth/elevation.
        scores = compute_v5_scores(
            image_width=W, image_height=H,
            bbox_full=tuple(bbox_new),
            azimuth_deg=float(entry.get("azimuth_deg", 0.0)),
            elevation_deg=float(entry.get("elevation_deg", 0.0)),
        )
        for k, v in scores.items():
            entry[f"score_{k}"] = v
        updated += 1

    ann_path.write_text(json.dumps(ann, indent=2))
    return {"skipped": False, "n_total": len(ann), "n_updated": updated}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python apply_mesh_tight_to_annotations.py <placement_dir> [<placement_dir> ...]", flush=True)
        sys.exit(1)
    for t in args:
        p = Path(t).resolve()
        if not p.is_dir():
            print(f"[skip] {p}", flush=True)
            continue
        r = process_placement(p)
        if r["skipped"]:
            print(f"[skip] {p.name}: {r['reason']}", flush=True)
        else:
            print(f"[ok]   {p.name}: {r['n_updated']}/{r['n_total']} entries updated", flush=True)


if __name__ == "__main__":
    main()
