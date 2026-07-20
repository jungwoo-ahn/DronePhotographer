"""Regenerate the held-out validation scene manifest.

Scene-level split for unseen-scene generalization: hold out the SMALLEST scenes
(cheapest in training data) until we cross `--frac` of scored placements. Scenes
sharing an asset family (e.g. the two Forest-field IDs) are kept together to
avoid lookalike leakage. Deterministic — rerun when the dataset changes.

  python scripts/make_val_split.py \
      --root data/trajectories_full --out configs/policy/val_scenes.yaml --frac 0.05
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
    p.add_argument("--frac", type=float, default=0.05, help="target val fraction of scored placements")
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

    val_scenes: list[str] = []
    acc = 0
    for fam in sorted(fam_size, key=lambda f: (fam_size[f], f)):
        val_scenes.extend(sorted(fam_scenes[fam]))
        acc += fam_size[fam]
        if acc >= a.frac * total:
            break
    val_scenes.sort()

    doc = {
        "description": "Held-out validation scenes (unseen-scene generalization). "
                       "Smallest scenes by scored-placement count; asset families kept together. "
                       "Regenerate with scripts/make_val_split.py when the dataset changes.",
        "root": str(a.root),
        "generated": a.date,
        "total_scenes": len(counts),
        "total_scored_placements": total,
        "val_placements": acc,
        "val_fraction": round(acc / total, 4),
        "scenes": [{"name": s, "placements": counts[s]} for s in val_scenes],
    }
    a.out.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    print(f"wrote {a.out}: {len(val_scenes)} scenes, {acc}/{total} placements ({acc/total*100:.1f}%)")
    for s in val_scenes:
        print(f"  {counts[s]:4d}  {s}")


if __name__ == "__main__":
    main()
