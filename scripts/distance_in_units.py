"""What the rollout distance is, in degrees, metres and percent.

`d` from `src/common/reward.py` is a weight-free product metric in radians:

    d = sqrt( d_view^2 + d_size^2 + d_aim^2 )

That is the right thing to RANK by — every axis carries equal weight and no unit conversion
smuggles in a preference. It is the wrong thing to READ. "0.25 radians from the requested shot"
tells a photographer nothing, and the three terms it collapses are three different questions:
the camera is in the wrong PLACE, the subject is the wrong SIZE, it sits in the wrong part of
the FRAME. A policy can be excellent at one and hopeless at another and the scalar will not say.

This decomposes it from `pose_trace` + `goal_pose`, which `closed_loop_eval.py` now records, so
no rollout is re-run:

| reported          | unit | what it is                                                       |
|-------------------|------|------------------------------------------------------------------|
| `view_deg`        | deg  | great-circle angle between the two viewing directions around the  |
|                   |      | subject -- "the camera is 23 degrees around the wrong way"        |
| `aim_deg`         | deg  | angle between the optical axis and the subject direction          |
| `occupancy_pp`    | pp   | difference in the fraction of frame HEIGHT the subject spans,     |
|                   |      | in percentage points -- the same quantity the goal profile names  |
| `range_err_m`     | m    | signed camera-to-subject distance error (+ = too far away)        |
| `pos_err_m`       | m    | straight-line distance between the two camera positions           |
| `height_err_m`    | m    | camera height error, the axis a dolly-only policy cannot fix      |

`occupancy_pp` is the one worth explaining. `d_size` is a difference of half-angles,
`atan((h/2)/r)`; the picture-level quantity is what fraction of the frame the subject fills,
which for a 24 mm lens on a 12.8 x 9.6 mm sensor (`BlenderDrone`) is `2*atan((h/2)/r) / vfov`.
Converting through the real lens rather than quoting the raw radian is what makes "12 points
too small" mean the thing a photographer would check.

    .venv-analysis/bin/python scripts/distance_in_units.py --runs runs/closed_loop/cmp_*.json
venv: .venv-analysis (pure numpy, instant).
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

# BlenderDrone's camera: 24 mm on a 12.8 x 9.6 mm sensor. The vertical half-angle is what turns
# an angular subject size into a fraction of the frame.
FOCAL_MM, SENSOR_H_MM = 24.0, 9.6
VFOV_RAD = 2.0 * math.atan((SENSOR_H_MM / 2.0) / FOCAL_MM)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default=f"{V12}/runs/eval/distance_units.json")
args = ap.parse_args()


def decompose(pose: dict, goal_pose: dict, center, height: float) -> dict:
    """One achieved pose against the goal, in readable units."""
    a = pose_to_geometry(pose["position"], pose["forward"], pose["up"], center, height)
    g = pose_to_geometry(goal_pose["position"], goal_pose["forward"], goal_pose["up"], center, height)

    # view: great-circle on the (azimuth, elevation) sphere -- the same term reward.py uses
    ca, cg = math.cos(a["el"]), math.cos(g["el"])
    cosang = (math.sin(a["el"]) * math.sin(g["el"]) + ca * cg * math.cos(a["az"] - g["az"]))
    view = math.acos(max(-1.0, min(1.0, cosang)))

    # size: half-angle -> fraction of frame height -> percentage points
    occ = lambda gm: 2.0 * gm["size"] / VFOV_RAD          # noqa: E731
    pa, pg = np.asarray(pose["position"], float), np.asarray(goal_pose["position"], float)
    c = np.asarray(center, float)
    return {
        "view_deg": math.degrees(view),
        "aim_deg": math.degrees(math.hypot(a["aim_x"] - g["aim_x"], a["aim_y"] - g["aim_y"])),
        "occupancy_pp": 100.0 * (occ(a) - occ(g)),
        "range_err_m": float(np.linalg.norm(pa - c) - np.linalg.norm(pg - c)),
        "pos_err_m": float(np.linalg.norm(pa - pg)),
        "height_err_m": float(pa[2] - pg[2]),
        "d_rad": math.sqrt(view ** 2 + (a["size"] - g["size"]) ** 2
                           + (a["aim_x"] - g["aim_x"]) ** 2 + (a["aim_y"] - g["aim_y"]) ** 2),
    }


def main() -> int:
    runs, need_rerun = {}, []
    for p in args.runs:
        d = json.loads(Path(p).read_text())
        eps = d["episodes"]
        if not eps or "pose_trace" not in eps[0]:
            need_rerun.append(Path(p).stem.replace("cmp_", ""))
            continue
        runs[Path(p).stem.replace("cmp_", "")] = eps
    if need_rerun:
        print(f"no pose_trace in: {', '.join(need_rerun)} — those runs predate the pose logging "
              f"and must be re-run; nothing else can recover the final pose.", file=sys.stderr)
    if not runs:
        return 1

    shared = sorted(set.intersection(*(set(e["key"] for e in v) for v in runs.values())))
    print(f"{len(runs)} policies, {len(shared)} shared episodes\n", flush=True)

    fields = ("view_deg", "aim_deg", "occupancy_pp", "range_err_m", "pos_err_m", "height_err_m")
    out = {}
    hdr = f"{'policy':10} {'d(rad)':>7} " + " ".join(f"{f.split('_')[0]:>9}" for f in fields)
    print(hdr); print("-" * len(hdr))
    for pol, eps in runs.items():
        by = {e["key"]: e for e in eps}
        rows = [decompose(by[k]["pose_trace"][-1], by[k]["goal_pose"],
                          by[k]["subject_center"], by[k]["subject_height"]) for k in shared]
        # Median, not mean: these are error magnitudes with a heavy right tail, and one episode
        # that flew across the room should not set the headline for the other 53.
        med = {f: float(np.median([abs(r[f]) if f in ("range_err_m", "height_err_m", "occupancy_pp")
                                   else r[f] for r in rows])) for f in fields}
        med["d_rad"] = float(np.median([r["d_rad"] for r in rows]))
        # signed medians say WHICH WAY a policy is wrong, which the absolute value hides
        signed = {f: float(np.median([r[f] for r in rows]))
                  for f in ("occupancy_pp", "range_err_m", "height_err_m")}
        out[pol] = {"median_abs": med, "median_signed": signed, "n": len(rows)}
        print(f"{pol:10} {med['d_rad']:>7.3f} " + " ".join(f"{med[f]:>9.2f}" for f in fields))

    print(f"\nsigned medians (which way the policy is wrong; + = too far / too big / too high)")
    print(f"{'policy':10} {'occupancy_pp':>13} {'range_m':>9} {'height_m':>9}")
    for pol, v in out.items():
        s = v["median_signed"]
        print(f"{pol:10} {s['occupancy_pp']:>13.2f} {s['range_err_m']:>9.2f} {s['height_err_m']:>9.2f}")

    print(f"\nlens: {FOCAL_MM:.0f} mm on {SENSOR_H_MM:.1f} mm sensor height "
          f"-> vertical FOV {math.degrees(VFOV_RAD):.1f} deg (BlenderDrone)")
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "_what": "The rollout distance decomposed into readable units. Medians over the shared "
                 "episodes; `median_abs` is the typical error magnitude, `median_signed` says "
                 "which way it leans.",
        "vfov_deg": math.degrees(VFOV_RAD), "episodes": len(shared), "policies": out}, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
