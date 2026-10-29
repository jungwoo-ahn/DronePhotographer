"""Merge sharded closed-loop runs into one summary.json.

A 64-episode PhotoFlow run is ~19 h on one GPU, so it is split across GPUs by episode and
stitched back together here. Aggregates are recomputed from the pooled `results` with the
same formulas as `closed_loop_eval_baselines.main`, never averaged across shards (a mean of
per-shard means is wrong whenever the shards differ in size).

  python scripts/merge_shard_summaries.py --out runs/shared/photoflow \
      runs/shared/photoflow_s0 runs/shared/photoflow_s1 runs/shared/photoflow_s2
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=None, help="refuse to merge unless n matches")
    a = ap.parse_args()

    results, meta, summed = [], None, {"api_calls": 0, "api_failures": 0}
    for sd in a.shards:
        f = Path(sd) / "summary.json"
        if not f.exists():
            raise SystemExit(f"missing {f} -- shard did not finish; refusing to merge a partial run")
        s = json.loads(f.read_text())
        results.extend(s["results"])
        for k in summed:
            summed[k] += (s.get(k) or 0)
        if meta is None:
            meta = s

    keys = [r["key"] for r in results]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise SystemExit(f"{len(dupes)} episode(s) appear in more than one shard, e.g. "
                         f"{sorted(dupes)[:3]} -- shards overlap; refusing to merge")
    if a.expect is not None and len(results) != a.expect:
        raise SystemExit(f"merged {len(results)} episodes, expected {a.expect}; refusing")

    imp = np.array([r["improvement"] for r in results], dtype=float)
    best = np.array([r["best_improvement"] for r in results], dtype=float)
    out = {**{k: v for k, v in meta.items() if k not in ("results",)},
           "n_rollouts": len(results),
           "mean_improvement_over_noop": float(imp.mean()),
           "median_improvement": float(np.median(imp)),
           "frac_positive": float((imp > 0).mean()),
           "mean_best_improvement": float(best.mean()),
           "frac_best_positive": float((best > 0).mean()),
           "mean_d_start": float(np.mean([r["d_start"] for r in results])),
           "mean_d_end": float(np.mean([r["d_end"] for r in results])),
           "success_at": {f"{t}": {"end": float(np.mean([r["d_end"] <= t for r in results])),
                                   "best": float(np.mean([r["d_best"] <= t for r in results]))}
                          for t in (0.05, 0.10, 0.20)},
           **summed,
           "merged_from": [str(s) for s in a.shards],
           "results": sorted(results, key=lambda r: r["key"] or ""),
           }
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"merged {len(results)} episodes from {len(a.shards)} shards -> {d/'summary.json'}")
    print(f"mean improvement {out['mean_improvement_over_noop']:+.4f}  "
          f"frac+ {out['frac_positive']:.3f}  d_end {out['mean_d_end']:.4f}")


if __name__ == "__main__":
    main()
