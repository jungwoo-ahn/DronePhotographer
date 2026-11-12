"""Merge PhotoFlow shard rollouts into runs/bench_v2/rollout_photoflow.json.

    python scripts/merge_photoflow_shards.py [--dir runs/bench_v2/photoflow_shards]

Finished shards are preferred over their `.partial.json`, so this can also be run mid-way to
see how far the run is and what it has cost. Roots follow DP_V12_ROOT like the harness.

Back on the cluster, after copying the photoflow_shards folder into runs/bench_v2/:
    python scripts/merge_photoflow_shards.py --rebase-from /path/on/the/other/server/DronePhotographerV12
"""
import argparse
import json
import os
from pathlib import Path

V12 = Path(os.environ.get("DP_V12_ROOT", "/home/nas_main/jungwooahn/projects/DronePhotographerV12"))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(V12 / "runs/bench_v2/photoflow_shards"))
ap.add_argument("--manifest", default=str(V12 / "configs/eval_episodes_v2.json"))
ap.add_argument("--out", default=str(V12 / "runs/bench_v2/rollout_photoflow.json"))
ap.add_argument("--rebase-from", default="", metavar="OLD_ROOT",
                help="the DP_V12_ROOT the shards were RUN under, when merging on another machine: "
                     "every path under it is rewritten to this machine's root, so the scorers "
                     "find the final frames")
args = ap.parse_args()

D = Path(args.dir)
order = [e["key"] for e in json.loads(Path(args.manifest).read_text())["episodes"]]
eps, summary, partial = {}, None, []
stems = sorted({p.name.split(".")[0] for p in D.glob("rollout_photoflow_s*.json")})
for stem in stems:
    p = D / f"{stem}.json"
    src = p if p.exists() else D / f"{stem}.partial.json"
    if src != p:
        partial.append(stem)
    text = src.read_text()
    if args.rebase_from:
        text = text.replace(args.rebase_from.rstrip("/") + "/", str(V12) + "/")
    d = json.loads(text)
    summary = summary or d.get("summary")
    eps.update({e["key"]: e for e in d["episodes"]})
out = [eps[k] for k in order if k in eps]
imp = [e["improvement"] for e in out]
recs = [e.get("policy_record") or {} for e in out]
summary = {**(summary or {}), "episodes": len(out), "attempted": len(order),
           "failed": len(order) - len(out), "merged_from": str(D),
           "mean_improvement_over_noop": sum(imp) / max(1, len(imp)),
           "mean_d_end": sum(e["d_end"] for e in out) / max(1, len(out)),
           "cost_usd": round(sum(r.get("cost_usd", 0) for r in recs), 2),
           "vlm_calls": sum(r.get("n_vlm_calls", 0) for r in recs)}
print(f"{len(out)}/{len(order)} episodes from {len(stems)} shards (unfinished: {partial or 'none'}); "
      f"cost ${summary['cost_usd']}, {summary['vlm_calls']} calls, mean d_end {summary['mean_d_end']:.3f}")
if len(out) == len(order):
    Path(args.out).write_text(json.dumps({"summary": summary, "episodes": out}))
    print(f"wrote {args.out}")
else:
    print("not all episodes are in yet; nothing written (rerun the launcher to resume)")
