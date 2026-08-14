"""Regenerate AutoPhoto's TRAIN placement manifest against the scene-level val split.

AutoPhoto (RL) trains on an explicit list of placement `data.json` paths rather than
the window dataset, so its train/val split is a materialized file. This regenerates it
to match the V12 scene-level val (`configs/policy/val_scenes.json`): every SCORED
placement (stage-3 done) whose scene is NOT one of the held-out val scenes.

  python scripts/make_autophoto_train_placements.py \
      --root data/trajectories_full \
      --val-scenes configs/policy/val_scenes.json \
      --out configs/policy/autophoto_train_placements.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml


def _scene(data_json: str) -> str:
    """Scene identity of a placement's data.json = its parent dir name before '__'
    (matches BasePolicyDataset _split_key level=scene)."""
    return Path(data_json).parent.name.split("__")[0]


def _is_scored(data_json: str) -> bool:
    """Stage-3 done: has per-frame scores (== stage3_n_frames > 0)."""
    try:
        d = json.loads(Path(data_json).read_text())
    except Exception:
        return False
    if int(d.get("stage3_n_frames") or 0) > 0:
        return True
    for rr in (d.get("render_records") or []):
        recs = rr if isinstance(rr, list) else [rr]
        if any((r.get("scores") or {}) for r in recs):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/trajectories_full")
    ap.add_argument("--val-scenes", type=Path, default=Path("configs/policy/val_scenes.json"))
    ap.add_argument("--out", type=Path, default=Path("configs/policy/autophoto_train_placements.yaml"))
    args = ap.parse_args()

    val = json.loads(args.val_scenes.read_text())
    val_scenes = set(val["scenes"])
    assert val.get("level") == "scene", f"expected scene-level manifest, got {val.get('level')}"

    all_json = sorted(glob.glob(f"{args.root}/*/data.json"))
    scored = [p for p in all_json if _is_scored(p)]
    train = [p for p in scored if _scene(p) not in val_scenes]
    dropped = len(scored) - len(train)

    doc = {
        "description": (
            f"AutoPhoto TRAIN placements: scored (stage-3), scene NOT in "
            f"{args.val_scenes.as_posix()} (V12 scene-level val, {len(val_scenes)} held-out scenes)."
        ),
        "root": args.root,
        "n_placements": len(train),
        "placements": train,
    }
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=1000))
    print(f"total data.json: {len(all_json)}  scored: {len(scored)}  "
          f"val-scene (dropped): {dropped}  ->  TRAIN: {len(train)}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
