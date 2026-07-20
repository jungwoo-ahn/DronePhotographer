"""Regenerate the held-out validation scene manifest.

Scene-level split for unseen-scene generalization, kept MINIMAL: hold out the
FEWEST (smallest) scenes that still give a statistically stable val set. Each
placement yields ~`--windows-per-placement` windows, so we add the smallest
scenes until the val set reaches `--min-windows` — then stop. This spends as
little scene diversity + training data as possible. Scenes sharing an asset
family (e.g. the two Forest-field IDs) are kept together to avoid lookalike
leakage. Deterministic — rerun when the dataset changes.

  python scripts/make_val_split.py \
      --root data/trajectories_full --out configs/policy/val_scenes.yaml --min-windows 1500
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def scene_of(placement: str) -> str:
    return placement.split("__")[0]


def family_of(scene: str) -> str:
    # Group scenes whose name (minus the trailing _<hash>) matches — e.g. the two
    # Forest-field_<id> scenes share family "Forest-field" and must not straddle
    # the train/val boundary.
    return scene.rsplit("_", 1)[0] if "_" in scene else scene


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/trajectories_full", type=Path)
    p.add_argument("--out", default="configs/policy/val_scenes.yaml", type=Path)
    p.add_argument("--min-windows", type=int, default=1500,
                   help="stop once the val set reaches this many windows (minimal split)")
    p.add_argument("--windows-per-placement", type=int, default=20,
                   help="approx windows per placement (32-frame traj, chunk 8, stride ~1)")
    p.add_argument("--train-out", default=None, type=Path,
                   help="also write the AutoPhoto train placement list (scored, not in val) here")
    p.add_argument("--date", default=None, help="stamp for provenance (YYYY-MM-DD); avoids nondeterminism")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    counts: Counter[str] = Counter()
    for d in sorted(a.root.iterdir()):
        if d.is_dir() and (d / "scored.flag").exists():
            counts[scene_of(d.name)] += 1
    total = sum(counts.values())

    # family size = sum over member scenes; order families by size then name.
    fam_scenes: dict[str, list[str]] = defaultdict(list)
    for s in counts:
        fam_scenes[family_of(s)].append(s)
    fam_size = {f: sum(counts[s] for s in ss) for f, ss in fam_scenes.items()}

    min_placements = -(-a.min_windows // max(1, a.windows_per_placement))  # ceil
    val_scenes: list[str] = []
    acc = 0
    for fam in sorted(fam_size, key=lambda f: (fam_size[f], f)):
        val_scenes.extend(sorted(fam_scenes[fam]))
        acc += fam_size[fam]
        if acc >= min_placements:
            break
    val_scenes.sort()

    doc = {
        "description": "Held-out validation scenes (unseen-scene generalization). Minimal: "
                       "fewest smallest scenes reaching ~min_windows; asset families kept "
                       "together. Regenerate with scripts/make_val_split.py when the data changes.",
        "root": str(a.root),
        "generated": a.date,
        "total_scenes": len(counts),
        "total_scored_placements": total,
        "val_scenes": len(val_scenes),
        "val_placements": acc,
        "val_fraction": round(acc / total, 4),
        "approx_val_windows": acc * a.windows_per_placement,
        "scenes": [{"name": s, "placements": counts[s]} for s in val_scenes],
    }
    a.out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    print(f"wrote {a.out}: {len(val_scenes)} scenes, {acc}/{total} placements ({acc/total*100:.1f}%)")
    for s in val_scenes:
        print(f"  {counts[s]:4d}  {s}")

    # Optionally materialize the TRAIN placement list (scored, NOT in val) —
    # AutoPhoto's RL env iterates an explicit list rather than splitting on the fly.
    if a.train_out:
        val_set = set(val_scenes)
        train = [f"{a.root}/{d.name}/data.json" for d in sorted(a.root.iterdir())
                 if d.is_dir() and (d / "scored.flag").exists() and scene_of(d.name) not in val_set]
        tdoc = {
            "description": "AutoPhoto TRAIN placements: every scored placement NOT in the "
                           "val scenes (configs/policy/val_scenes.yaml). Regenerate with "
                           "scripts/make_val_split.py --train-out when the data changes.",
            "root": str(a.root),
            "generated": a.date,
            "n_placements": len(train),
            "placements": train,
        }
        Path(a.train_out).write_text(yaml.safe_dump(tdoc, sort_keys=False, default_flow_style=False))
        print(f"wrote {a.train_out}: {len(train)} train placements (val scenes excluded)")


if __name__ == "__main__":
    main()
