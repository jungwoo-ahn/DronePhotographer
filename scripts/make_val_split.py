"""Generate a leak-free validation split.

Given the v7 data structure — 101 subjects each rendered across ~all 46 scenes
(mean 38.5 scenes/subject) — scene and object are almost fully crossed, so a
strict scene-AND-object-disjoint split is infeasible (it would discard ~89% of
placements). We hold out one axis cleanly instead (0 discard):

  object  (default) — hold out whole SUBJECTS: val = every placement of the
          held-out subjects (across shared scenes); train = all other subjects.
          Measures generalization to unseen subjects — the axis a framing policy
          must generalize on, and the one the old scene-level split leaked.
  scene   — hold out whole SCENES (old val_scenes.txt semantics); objects stay
          shared across train/val.

Identity is name-level (trailing `_<8hex>` asset hash stripped) so the same
asset under two hashes is one unit and never straddles the split.

Output (consumed via `val_split_level: placement`, `val_names: <file>`):
  * val_placements_v7.txt — explicit val placement dir names.
Train is everything not listed; neither axis produces crossover to exclude.

Usage:
  PYTHONPATH=. python scripts/make_val_split.py --mode object --n-units 5
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

HASH_SUFFIX = re.compile(r"_[0-9a-fA-F]{8}$")


def scene_name(placement: str) -> str:
    """Name-level scene identity: '<Scene>_<hash>__<Obj>_<hash>' -> '<Scene>'."""
    return HASH_SUFFIX.sub("", placement.split("__")[0])


def object_name(placement: str) -> str:
    """Name-level object (subject) identity, hash stripped."""
    return HASH_SUFFIX.sub("", placement.split("__")[-1])


def enumerate_placements(root: Path) -> list[str]:
    """Placement dir names carrying a data.json (the set training sees).

    One-level scan of a known dir (no recursion) — safe on the shared NAS.
    """
    return sorted(p.name for p in root.iterdir() if (p / "data.json").exists())


def select_units(placements: list[str], key_fn, n_units: int) -> set[str]:
    """Pick `n_units` held-out units (scenes or subjects) in a stable pseudo-random
    order (md5 of the unit name), so the choice is representative and reproducible."""
    units = sorted({key_fn(p) for p in placements},
                   key=lambda u: hashlib.md5(u.encode()).hexdigest())
    if n_units >= len(units):
        raise SystemExit(f"n_units={n_units} >= total units={len(units)} — nothing left for train")
    return set(units[:n_units])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--mode", choices=["object", "scene"], default="object")
    ap.add_argument("--n-units", type=int, default=5, help="how many subjects (object) / scenes to hold out")
    ap.add_argument("--out-val", type=Path, default=Path("configs/policy/val_placements_v7.txt"))
    args = ap.parse_args()

    placements = enumerate_placements(args.root)
    total = len(placements)
    if not total:
        raise SystemExit(f"no placements with data.json under {args.root}")

    key_fn = object_name if args.mode == "object" else scene_name
    other_fn = scene_name if args.mode == "object" else object_name
    held = select_units(placements, key_fn, args.n_units)

    val = [p for p in placements if key_fn(p) in held]
    train = [p for p in placements if key_fn(p) not in held]

    # Disjointness on the held-out axis (the guarantee); the other axis is shared.
    val_units = {key_fn(p) for p in val}
    train_units = {key_fn(p) for p in train}
    assert not (val_units & train_units), f"{args.mode} overlap between val and train"
    assert len(val) + len(train) == total, "partition does not cover all placements"
    shared_other = len({other_fn(p) for p in val} & {other_fn(p) for p in train})
    other_label = "scenes" if args.mode == "object" else "subjects"

    header = (
        f"# leak-free val split (scripts/make_val_split.py) — mode={args.mode}\n"
        f"# total placements: {total}\n"
        f"# val:   {len(val):5d} ({100*len(val)/total:.1f}%)  holding out {len(held)} {args.mode}(s)\n"
        f"# train: {len(train):5d} ({100*len(train)/total:.1f}%)  (0 discard)\n"
        f"# held-out {args.mode}s: {', '.join(sorted(held))}\n"
        f"# val {args.mode}s are disjoint from train; the {other_label} axis is shared "
        f"({shared_other} {other_label} in both — expected for this data)\n"
    )
    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    args.out_val.write_text(header + "\n".join(val) + "\n")

    print(header)
    print(f"wrote {args.out_val}  ({len(val)} placements)")


if __name__ == "__main__":
    main()
