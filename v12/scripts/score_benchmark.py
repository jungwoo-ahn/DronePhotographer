"""Score closed-loop runs on a benchmark manifest into the paper's columns.

`closed_loop_eval.py` records the pose after every chunk and the pose-space distance `d` to the
goal. A photograph also has to show the subject, at the right size, in the right part of the
frame, and `d` alone does not say any of that: on the trajectory-GIF set one run ended 0.049 from
its goal on a black frame, its camera inside a wall. So each final pose is re-scored here:

  * composition, exactly: the stage-3 scorer over the subject's projected mesh
    (`PlacementGeometry.profile`, the same projection that labelled the training data) gives the
    occupancy and the subject's centre at the final pose; one Blender dump per placement.
  * visibility: a person detector on the final frame, and its box must overlap where the mesh
    projects. The mesh cannot see occlusion; the detector can.

Columns (one row per run, n = episodes the run finished):
  Succ / Strict   d <= 0.10 / 0.05 AND the subject visible, with a Wilson 95% interval
  View deg        angle between the achieved and the target viewing directions around the subject
  Size pp         |occupancy - target occupancy|, percentage points of the frame area (median)
  Frame %         distance between the subject's centre and the target's, % of frame width (median)
  Visible %       final frames in which the detector finds the subject
  p vs ref        exact McNemar test on Succ against --ref over the episodes both finished

    python scripts/score_benchmark.py geom   --manifest configs/eval_episodes_v2.json   # Blender, CPU
    python scripts/score_benchmark.py detect --runs ours=runs/bench_v2/ours_v6_45k.json ...   # GPU
    python scripts/score_benchmark.py table  --runs ours=... i2a=... --ref ours
venv: .venv-analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=("geom", "detect", "table"))
ap.add_argument("--manifest", default=f"{V12}/configs/eval_episodes_v2.json")
ap.add_argument("--dir", default=f"{V12}/runs/bench_v2")
ap.add_argument("--runs", nargs="*", default=[], metavar="LABEL=JSON")
ap.add_argument("--ref", default="ours")
ap.add_argument("--perceptual", default="", help="perceptual_eval.py output, for an Img sim column")
ap.add_argument("--tau", type=float, nargs=2, default=(0.10, 0.05), metavar=("SUCC", "STRICT"))
ap.add_argument("--visible-iou", type=float, default=0.3)
ap.add_argument("--workers", type=int, default=8, help="parallel Blender dumps")
ap.add_argument("--blender", default=f"{SHARED}/blender/blender")
args = ap.parse_args()

D = Path(args.dir)
GEOM = D / "_geom"


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else Path(V12) / p


def runs() -> dict[str, dict]:
    """label -> {key: episode}, finished file preferred over its partial."""
    out = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        p = _abs(path)
        src = p if p.exists() else p.with_suffix(".partial.json")
        out[label] = {e["key"]: e for e in json.loads(src.read_text())["episodes"]}
    return out


def stage_geom(manifest: dict) -> int:
    from src.common.run_info import write_run_info

    GEOM.mkdir(parents=True, exist_ok=True)
    todo = {}
    for e in manifest["episodes"]:
        if not (GEOM / f"{e['placement']}.npz").exists():
            todo[e["placement"]] = e["data_json"]
    print(f"[geom] {len(todo)} placements to dump ({args.workers} at a time)", flush=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{SHARED}/blender/syslibs/lib:" + env.get("LD_LIBRARY_PATH", "")

    def dump(item):
        pl, data = item
        ri = write_run_info(pl, data, D / "_geom_run_info", shared_root=SHARED, resolution=(1024, 768))
        npz = GEOM / f"{pl}.npz"
        r = subprocess.run([args.blender, "-b", "-P", f"{V12}/scripts/dump_placement_geometry.py",
                            "--", "--run-info", ri, "--data", data, "--out", str(npz)],
                           cwd=SHARED, capture_output=True, text=True, timeout=900, env=env)
        return pl, npz.exists(), r.stderr[-800:]

    t0, failed = time.time(), []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (pl, ok, err) in enumerate(pool.map(dump, sorted(todo.items()))):
            if not ok:
                failed.append(pl)
                print(f"[geom] FAILED {pl}: {err}", flush=True)
            if (i + 1) % 20 == 0:
                print(f"[geom] {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[geom] done, {len(failed)} failed ({time.time() - t0:.0f}s)")
    return 1 if failed else 0


def stage_detect() -> int:
    from src.common.visibility import detect_people

    out_path = D / "final_detect.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    frames = []
    for label, eps in runs().items():
        for key, r in eps.items():
            frames.append(r["shots"][-1]["path"])
    todo = [f for f in dict.fromkeys(frames) if f not in out]
    print(f"[detect] {len(todo)} new final frames ({len(out)} cached)", flush=True)
    out.update(detect_people(todo))
    out_path.write_text(json.dumps(out))
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, 100 * (mid - half)), min(100.0, 100 * (mid + half)))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar: b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score_episode(ep: dict, r: dict, geom, det: dict) -> dict:
    from src.common.visibility import best_iou, clip_box

    pf = r["pose_trace"][-1]
    pos, fwd, up = (np.asarray(pf[k], dtype=np.float64) for k in ("position", "forward", "up"))
    gp = ep["goal_pose"]
    c = np.asarray(ep["subject_center"], dtype=np.float64)
    v_f, v_g = pos - c, np.asarray(gp["position"]) - c
    view = math.degrees(math.acos(float(np.clip(
        np.dot(v_f, v_g) / (np.linalg.norm(v_f) * np.linalg.norm(v_g)), -1.0, 1.0))))
    # The float twin of the stage-3 scorer: the integer one rounds occupancy to whole percent,
    # which collapsed every good policy's median size error onto exactly 1.0.
    goal = geom.profile_float(np.asarray(gp["position"]), np.asarray(gp["forward"]), np.asarray(gp["up"]))
    fin = geom.profile_float(pos, fwd, up)
    box = geom.full_bbox(pos, fwd, up)
    vis_box = clip_box(box, geom.width, geom.height) if box is not None else None
    final = r["shots"][-1]["path"]
    if final not in det:
        # Never default to "not visible": a frame the detector has not seen is unscored, and
        # counting it as a miss is how a stale detect pass quietly lowers a run's Succ.
        raise SystemExit(f"no detection for {final}: re-run the detect stage on these runs")
    visible = bool(vis_box is not None and best_iou(det[final], vis_box) >= args.visible_iou)
    if fin is None:                       # the subject projects entirely outside the frame
        size, frame = goal["occupancy"], 100.0
    else:
        size = abs(fin["occupancy"] - goal["occupancy"])
        frame = 100.0 * math.hypot(fin["object_center_x"] - goal["object_center_x"],
                                   fin["object_center_y"] - goal["object_center_y"]) / geom.width
    d = float(r["d_end"])
    return {"d_end": d, "d_start": float(r["d_start"]), "visible": visible,
            "succ": d <= args.tau[0] and visible, "strict": d <= args.tau[1] and visible,
            "view_deg": view, "size_pp": size, "frame_pct": frame,
            "declared": r.get("declared_stop") is not None,
            "start_render_mae": r.get("start_render_mae")}


def stage_table(manifest: dict) -> int:
    from src.goal_authoring.pose_solver import PlacementGeometry

    eps = {e["key"]: e for e in manifest["episodes"]}
    geoms = {p.stem: PlacementGeometry.load(p) for p in GEOM.glob("*.npz")}
    det_path = D / "final_detect.json"
    det = json.loads(det_path.read_text()) if det_path.exists() else {}
    # perceptual_eval.py keys its policies by run-file stem (`rollout_ours_v6_45k`).
    perc = {}
    if args.perceptual:
        per = json.loads(_abs(args.perceptual).read_text())["metrics"]["dinov3"]["per_policy"]
        for spec in args.runs:
            label, path = spec.split("=", 1)
            stem = Path(path).stem.replace("cmp_", "")
            if stem in per:
                perc[label] = per[stem]["end"]
    rows, scores = {}, {}
    for label, res in runs().items():
        sc = {}
        for key, r in res.items():
            if key not in eps:
                continue
            g = geoms.get(eps[key]["placement"])
            if g is None:
                raise SystemExit(f"no geometry for {eps[key]['placement']}: run the geom stage")
            sc[key] = score_episode(eps[key], r, g, det)
        scores[label] = sc
    ref = scores.get(args.ref, {})
    for label, sc in scores.items():
        n = len(sc)
        k_s = sum(v["succ"] for v in sc.values())
        k_t = sum(v["strict"] for v in sc.values())
        row = {"n": n, "attempted": len(eps),
               "succ": 100 * k_s / n, "succ_ci": wilson(k_s, n),
               "strict": 100 * k_t / n, "strict_ci": wilson(k_t, n),
               "view_deg": float(np.median([v["view_deg"] for v in sc.values()])),
               "size_pp": float(np.median([v["size_pp"] for v in sc.values()])),
               "frame_pct": float(np.median([v["frame_pct"] for v in sc.values()])),
               "visible": 100 * float(np.mean([v["visible"] for v in sc.values()])),
               "mean_d_end": float(np.mean([v["d_end"] for v in sc.values()])),
               "improvement": float(np.mean([v["d_start"] - v["d_end"] for v in sc.values()])),
               "declared": 100 * float(np.mean([v["declared"] for v in sc.values()])),
               "render_parity_over_6": sum(1 for v in sc.values()
                                           if (v["start_render_mae"] or 0) > 6.0)}
        if label != args.ref and ref:
            both = set(sc) & set(ref)
            b = sum(1 for k in both if sc[k]["succ"] and not ref[k]["succ"])
            c = sum(1 for k in both if ref[k]["succ"] and not sc[k]["succ"])
            row.update({"vs_ref_wins": b, "vs_ref_losses": c, "vs_ref_p": mcnemar(b, c)})
        if label in perc:
            row["img_sim"] = perc[label]
        rows[label] = row
    (D / "scores.json").write_text(json.dumps(scores, indent=1))
    (D / "table.json").write_text(json.dumps(rows, indent=1))
    hdr = (f"| run | n | Succ ↑ (95% CI) | Strict ↑ | View° ↓ | Size pp ↓ | Frame % ↓ | Visible ↑ "
           f"| Img sim ↑ | d_end | vs {args.ref} (W/L, p) |")
    lines = [hdr, "|" + "---|" * 11]
    for label, r in rows.items():
        vs = (f"{r['vs_ref_wins']}/{r['vs_ref_losses']}, {r['vs_ref_p']:.2g}"
              if "vs_ref_p" in r else "—")
        sim = f"{r['img_sim']:.3f}" if "img_sim" in r else "—"
        lines.append(f"| {label} | {r['n']}/{r['attempted']} | {r['succ']:.1f} "
                     f"({r['succ_ci'][0]:.0f}–{r['succ_ci'][1]:.0f}) | {r['strict']:.1f} | "
                     f"{r['view_deg']:.1f} | {r['size_pp']:.1f} | {r['frame_pct']:.1f} | "
                     f"{r['visible']:.0f} | {sim} | {r['mean_d_end']:.3f} | {vs} |")
    (D / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    manifest = json.loads(_abs(args.manifest).read_text())
    raise SystemExit({"geom": lambda: stage_geom(manifest), "detect": stage_detect,
                      "table": lambda: stage_table(manifest)}[args.stage]())
