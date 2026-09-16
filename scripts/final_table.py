"""One table: geometry, the same distance in readable units, and what the picture looks like.

Three scripts answer three different questions about the same rollouts and until now they
printed three tables:

  closed_loop_eval.py   how far from the requested shot, in radians -- the ranking metric
  distance_in_units.py  the same distance as degrees / metres / percentage points
  perceptual_eval.py    whether the frame it stopped on LOOKS like the frame that was asked for

Reading them side by side is where the findings are. `improvement` cannot separate our policy
from pi0.5; `success@0.05` separates them two to one. The perceptual metrics rank the top four
identically because they are saturated against the renderer gap, and say so only if the ceiling
is on the page next to them. The unit columns are what turn "0.25 radians" into the sentence a
photographer would act on.

So this joins all three on the episode key and prints one table, plus the paired statistics that
only a shared episode set allows.

    .venv-analysis/bin/python scripts/final_table.py --runs runs/closed_loop/cmp_*.json \\
        --perceptual runs/eval/perceptual.json --ref v6_12k
venv: .venv-analysis (the perceptual JSON is read, not recomputed).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)

from src.common.reward import pose_to_geometry  # noqa: E402

FOCAL_MM, SENSOR_H_MM = 24.0, 9.6
VFOV_RAD = 2.0 * math.atan((SENSOR_H_MM / 2.0) / FOCAL_MM)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--perceptual", default=f"{V12}/runs/eval/perceptual.json")
ap.add_argument("--thresholds", default=f"{V12}/runs/eval/shot_threshold.json")
ap.add_argument("--ref", default="v6_12k")
ap.add_argument("--order", default="", help="comma-separated policy order; default is by d_end")
ap.add_argument("--out", default=f"{V12}/runs/eval/final_table.json")
ap.add_argument("--markdown", default="", help="also write a markdown version here")
args = ap.parse_args()


def units(pose, goal_pose, center, height) -> dict:
    """The achieved pose against the goal, in degrees / metres / percentage points."""
    a = pose_to_geometry(pose["position"], pose["forward"], pose["up"], center, height)
    g = pose_to_geometry(goal_pose["position"], goal_pose["forward"], goal_pose["up"], center, height)
    cosang = (math.sin(a["el"]) * math.sin(g["el"])
              + math.cos(a["el"]) * math.cos(g["el"]) * math.cos(a["az"] - g["az"]))
    view = math.acos(max(-1.0, min(1.0, cosang)))
    occ = lambda m: 2.0 * m["size"] / VFOV_RAD          # noqa: E731
    pa, pg, c = (np.asarray(x, float) for x in (pose["position"], goal_pose["position"], center))
    return {"view_deg": math.degrees(view),
            "aim_deg": math.degrees(math.hypot(a["aim_x"] - g["aim_x"], a["aim_y"] - g["aim_y"])),
            "occ_pp": 100.0 * (occ(a) - occ(g)),
            "range_m": float(np.linalg.norm(pa - c) - np.linalg.norm(pg - c)),
            "pos_m": float(np.linalg.norm(pa - pg))}


def paired(a, b, seed=0):
    da = np.array([e["improvement"] for e in a], float)
    db = np.array([e["improvement"] for e in b], float)
    diff = da - db
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    wins, ties = int((diff > 0).sum()), int((diff == 0).sum())
    n = len(diff) - ties
    from math import comb
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)) if n else 1.0
    return {"mean_diff": float(diff.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": wins, "losses": n - wins, "sign_p": float(p)}


def main() -> int:
    runs = {}
    for p in args.runs:
        d = json.loads(Path(p).read_text())
        runs[Path(p).stem.replace("cmp_", "")] = d
    shared = sorted(set.intersection(*(set(e["key"] for e in d["episodes"]) for d in runs.values())))
    if not shared:
        print("no shared episodes", file=sys.stderr)
        return 1

    # d_start must agree: it is pure pose math, so a disagreement means different episodes.
    ref_ds = {k: next(e for e in next(iter(runs.values()))["episodes"] if e["key"] == k)["d_start"]
              for k in shared}
    for name, d in runs.items():
        for e in d["episodes"]:
            if e["key"] in ref_ds and abs(e["d_start"] - ref_ds[e["key"]]) > 1e-6:
                print(f"d_start mismatch on {e['key']} in {name} — different episodes", file=sys.stderr)
                return 1

    perc = {}
    if Path(args.perceptual).exists():
        perc = json.loads(Path(args.perceptual).read_text()).get("metrics", {})

    rows = {}
    for name, d in runs.items():
        by = {e["key"]: e for e in d["episodes"]}
        eps = [by[k] for k in shared]
        d_end = np.array([e["d_end"] for e in eps])
        d_best = np.array([e["d_best"] for e in eps])
        d_st = np.array([e["d_start"] for e in eps])
        imp = np.array([e["improvement"] for e in eps])
        row = {
            "n": len(eps),
            "d_end": float(d_end.mean()), "imp": float(imp.mean()),
            "median": float(np.median(imp)), "frac_pos": float((imp > 0).mean()),
            "closure": float(np.median(1.0 - d_end / np.maximum(d_st, 1e-9))),
            "s05": float((d_end <= 0.05).mean()), "s10": float((d_end <= 0.10).mean()),
            "s20": float((d_end <= 0.20).mean()), "best10": float((d_best <= 0.10).mean()),
            "declared": sum(1 for e in eps if e.get("declared_stop") is not None),
        }
        if all("pose_trace" in e for e in eps):
            u = [units(e["pose_trace"][-1], e["goal_pose"], e["subject_center"], e["subject_height"])
                 for e in eps]
            # medians: error magnitudes with a heavy right tail, so one runaway episode should
            # not set the headline for the other fifty-three
            row["units"] = {f: float(np.median([abs(x[f]) if f in ("occ_pp", "range_m") else x[f]
                                                for x in u]))
                            for f in ("view_deg", "aim_deg", "occ_pp", "range_m", "pos_m")}
        for m, mv in perc.items():
            if name in mv.get("per_policy", {}):
                row[f"perc_{m}"] = mv["per_policy"][name]["toward_goal"]
        rows[name] = row

    order = ([o.strip() for o in args.order.split(",") if o.strip()]
             or sorted(rows, key=lambda n: rows[n]["d_end"]))

    print(f"\n{len(rows)} policies, {len(shared)} shared episodes, "
          f"mean d_start {np.mean(list(ref_ds.values())):.3f}\n")

    has_u = any("units" in r for r in rows.values())
    has_p = any(k.startswith("perc_") for r in rows.values() for k in r)
    h = f"{'policy':11}{'d_end':>7}{'imp':>8}{'med':>7}{'frac+':>7}{'clos':>6}{'s@.05':>7}{'s@.10':>7}{'best@.10':>9}"
    if has_u:
        h += f"{'view°':>7}{'aim°':>6}{'occ%p':>7}{'range m':>8}{'pos m':>7}"
    if has_p:
        h += f"{'VGG':>6}{'CLIP':>6}{'DINO':>6}"
    print(h); print("-" * len(h))
    for n in order:
        r = rows[n]
        line = (f"{n:11}{r['d_end']:>7.3f}{r['imp']:>+8.3f}{r['median']:>7.3f}{r['frac_pos']:>7.2f}"
                f"{r['closure']:>6.0%}{r['s05']:>7.0%}{r['s10']:>7.0%}{r['best10']:>9.0%}")
        if has_u:
            u = r.get("units")
            line += ("".join(f"{u[f]:>{w}.1f}" for f, w in
                             (("view_deg", 7), ("aim_deg", 6), ("occ_pp", 7), ("range_m", 8), ("pos_m", 7)))
                     if u else f"{'—':>35}")
        if has_p:
            line += "".join(f"{r.get(f'perc_{m}', float('nan')):>6.2f}" for m in ("vgg", "clip", "dinov3"))
        print(line)

    if has_p:
        for m in ("vgg", "clip", "dinov3"):
            if m in perc:
                print(f"  {m} floor {perc[m]['floor']:.3f} -> ceiling {perc[m].get('ceiling', 1.0):.3f} "
                      f"(the ceiling is the SAME pose through the two renderers; the columns are "
                      f"normalised to it, so 1.00 is perfect, not similarity 1.0)")
                break

    if args.ref in rows:
        print(f"\npaired against {args.ref} (positive = {args.ref} closer on that episode):")
        print(f"{'policy':11}{'mean diff':>11}{'95% CI':>22}{'w/l':>9}{'sign p':>9}")
        base = [{"improvement": e["improvement"]} for e in
                sorted((e for e in runs[args.ref]["episodes"] if e["key"] in shared), key=lambda e: e["key"])]
        for n in order:
            if n == args.ref:
                continue
            other = [{"improvement": e["improvement"]} for e in
                     sorted((e for e in runs[n]["episodes"] if e["key"] in shared), key=lambda e: e["key"])]
            s = paired(base, other)
            ci = f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            print(f"{n:11}{s['mean_diff']:>+11.3f}{ci:>22}{f'{s['wins']}/{s['losses']}':>9}{s['sign_p']:>9.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": len(shared), "policies": rows,
                               "mean_d_start": float(np.mean(list(ref_ds.values())))}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
