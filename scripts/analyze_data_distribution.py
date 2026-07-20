"""v7 data distribution report — feeds the val split and sanity-checks the clamp fix.

Reports, over a sample of placements (one-level enumeration only — NAS-safe):
  * per-scene / per-object placement counts (name-level) + object sharing across scenes
  * multiscale window clamp rate by signed offset, BEFORE vs AFTER the recovery
    (before = baked sentinel in data.json; after = recovered in _frame_to_view)
  * dolly-in vs dolly-out balance and endpoint occupancy by offset
  * per-goal-key normalized distribution (mean / std → goal distinguishability)
  * per-frame mean luminance (flag near-black renders that can't supervise the world head)

  PYTHONPATH=. python scripts/analyze_data_distribution.py --sample 40 --lum-sample 15
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.policy.common.annotations import iter_multiscale_windows
from src.policy.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector, normalize_goal

HASH = re.compile(r"_[0-9a-fA-F]{8}$")
scene_of = lambda p: HASH.sub("", p.split("__")[0])
object_of = lambda p: HASH.sub("", p.split("__")[-1])


def _baked_clamped(rec: dict) -> bool:
    s = rec.get("scores") or {}
    return s.get("occupancy") == 0 and s.get("bbox_y_offset") == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--sample", type=int, default=40, help="placements sampled for window-level stats")
    ap.add_argument("--lum-sample", type=int, default=15, help="placements sampled for luminance")
    ap.add_argument("--dark-thresh", type=float, default=15.0, help="mean-pixel (0-255) below which a frame is 'near-black'")
    args = ap.parse_args()

    placements = sorted(p.name for p in args.root.iterdir() if (p / "data.json").exists())
    total = len(placements)
    scenes = Counter(scene_of(p) for p in placements)
    objects = Counter(object_of(p) for p in placements)
    obj_scenes = defaultdict(set)
    for p in placements:
        obj_scenes[object_of(p)].add(scene_of(p))
    spans = [len(s) for s in obj_scenes.values()]
    print(f"=== corpus === {total} placements | {len(scenes)} scenes | {len(objects)} objects")
    print(f"  scene size: min={min(scenes.values())} max={max(scenes.values())} mean={statistics.mean(scenes.values()):.1f}")
    print(f"  object appears in scenes: mean={statistics.mean(spans):.1f} max={max(spans)} "
          f"(shared>1: {sum(1 for x in spans if x>1)}/{len(objects)})")

    step = max(1, total // args.sample)
    samp = placements[::step][:args.sample]
    clamp = defaultdict(lambda: [0, 0, 0])   # offset -> [baked_clamped, still_clamped_after, total]
    occ_by_off = defaultdict(list)
    keyvals = defaultdict(list)
    ndolly_in = ndolly_out = 0
    for name in samp:
        dj = args.root / name / "data.json"
        import json
        recs = (json.loads(dj.read_text()).get("render_records") or [])
        baked_by = {}
        for i, rr in enumerate(recs):
            for r in rr:
                baked_by[(i, int(r.get("frame_idx", -1)))] = _baked_clamped(r)
        for w in iter_multiscale_windows(dj, chunk_size=8, offsets=(8, 16, 24)):
            off = w.end_frame_idx - w.start_frame_idx
            if off > 0: ndolly_in += 1
            else: ndolly_out += 1
            c = clamp[off]
            c[2] += 1
            if baked_by.get((w.pair_idx, w.end_frame_idx)):
                c[0] += 1
            raw = w.end.raw
            if raw.get("occupancy") == 0 and raw.get("bbox_y_offset") == 0:
                c[1] += 1
            oc = raw.get("occupancy_clipped")
            if oc is not None:
                occ_by_off[off].append(float(oc))
            g = normalize_goal(goal_vector(raw, DEFAULT_GOAL_KEYS), DEFAULT_GOAL_KEYS)
            if np.isfinite(g).all():
                for k, v in zip(DEFAULT_GOAL_KEYS, g):
                    keyvals[k].append(float(v))

    print(f"\n=== windows (sample {len(samp)}) === dolly-in={ndolly_in} dolly-out={ndolly_out} "
          f"({100*ndolly_in/max(1,ndolly_in+ndolly_out):.0f}% in)")
    print("clamp rate by signed offset  (baked -> after-recovery drop):")
    for off in sorted(clamp, key=lambda x: (abs(x), x)):
        b, a, t = clamp[off]
        occ = occ_by_off.get(off, [0])
        print(f"  off {off:+3d}: baked_clamped={100*b/max(1,t):4.1f}%  still_after={100*a/max(1,t):4.1f}%  "
              f"| endpoint occ mean={statistics.mean(occ):.2f}")

    print("\n=== goal-key normalized distribution (mean / std) ===")
    for k in DEFAULT_GOAL_KEYS:
        v = keyvals[k]
        print(f"  {k:26s} mean={statistics.mean(v):+.2f} std={statistics.pstdev(v):.2f}")

    # luminance
    from PIL import Image
    dark = tot = 0
    lums = []
    for name in placements[::max(1, total // args.lum_sample)][:args.lum_sample]:
        rr = None
        import json
        recs = json.loads((args.root / name / "data.json").read_text()).get("render_records") or []
        for i, frames in enumerate(recs[:1]):
            for r in frames[::8]:   # every 8th frame
                img = args.root / name / (r.get("path_rel") or "")
                if not img.exists(): continue
                m = float(np.asarray(Image.open(img).convert("L")).mean())
                lums.append(m); tot += 1
                if m < args.dark_thresh: dark += 1
    if tot:
        print(f"\n=== luminance (sample {tot} frames) === mean={statistics.mean(lums):.0f}/255  "
              f"near-black(<{args.dark_thresh:.0f}): {dark}/{tot} = {100*dark/tot:.0f}%")


if __name__ == "__main__":
    main()
