"""Pick the 3DGS benchmark episodes from `gs_build_episodes.py candidates` (paper Table 5).

The builder rendered a few candidate windows per placement. This applies benchmark v2's picture
rules to them -- the goal's subject box not dark, a person detected where the mesh projects in
the goal (IoU >= 0.5) and in the start (>= 0.3) -- and keeps --per-placement episodes from
different camera paths, dealt round-robin over the view sectors within each scene so no scene's
set collapses onto one side of the subject. No human pick.

Each scene gets --per-scene episodes. Where the reconstruction cannot support some of a scene's
placements (no camera path near enough to the capture cameras -- 5 of AG_0022's 10), the scene
is topped up from its other placements, at most --max-per-placement each and still one per
camera path, rather than letting the scene count for less in the table.

Writes `configs/eval_episodes_gs_v1.json` (kind "gs"), which `closed_loop_eval.py --episodes-from`
replays with the 3DGS renderer, sorted by placement so a run reloads each splat and each
subject's Blender as rarely as possible, plus contact sheets in runs/bench_gs.

    .venv-analysis/bin/python scripts/gs_select_episodes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)

import numpy as np  # noqa: E402

from src.common.visibility import best_iou, clip_box, detect_people, subject_luminance  # noqa: E402
from src.goal_authoring.pose_solver import PlacementGeometry  # noqa: E402

SECTORS = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=f"{V12}/runs/bench_gs")
ap.add_argument("--out", default=f"{V12}/configs/eval_episodes_gs_v1.json")
ap.add_argument("--per-placement", type=int, default=2)
ap.add_argument("--per-scene", type=int, default=20)
ap.add_argument("--max-per-placement", type=int, default=4, help="top-up cap for a short scene")
ap.add_argument("--min-lum", type=float, default=20.0)
ap.add_argument("--goal-iou", type=float, default=0.5)
ap.add_argument("--start-iou", type=float, default=0.3)
args = ap.parse_args()
D = Path(args.dir)


def box(geom: PlacementGeometry, pose: dict):
    b = geom.full_bbox(np.asarray(pose["position"]), np.asarray(pose["forward"]), np.asarray(pose["up"]))
    return clip_box(b, geom.width, geom.height) if b is not None else None


def main() -> int:
    # One file per builder shard (or one candidates.json); a key seen twice is the same window,
    # because each placement's paths are seeded by its name.
    report, cands, seen = {}, [], set()
    for f in sorted(D.glob("candidates*.json")):
        part = json.loads(f.read_text())
        report.update(part["report"])
        for c in part["candidates"]:
            if c["key"] not in seen:
                seen.add(c["key"])
                cands.append(c)
    print(f"[select] {len(cands)} candidate windows from {len(report)} placements", flush=True)
    geoms = {p.stem: PlacementGeometry.load(p) for p in (D / "_geom").glob("*.npz")}
    det = detect_people([c["goal_image"] for c in cands] + [c["start_image"] for c in cands])
    why, ok = Counter(), []
    for c in cands:
        g = geoms[c["placement"]]
        gb, sb = box(g, c["goal_pose"]), box(g, c["start_pose"])
        if gb is None or sb is None:
            why["off_frame"] += 1
            continue
        c["goal_detect_iou"] = round(best_iou(det[c["goal_image"]], gb), 3)
        c["start_detect_iou"] = round(best_iou(det[c["start_image"]], sb), 3)
        c["goal_subject_lum"] = round(subject_luminance(c["goal_image"], gb), 1)
        if c["goal_subject_lum"] < args.min_lum:
            why["dark_goal"] += 1
        elif c["goal_detect_iou"] < args.goal_iou:
            why["goal_not_detected"] += 1
        elif c["start_detect_iou"] < args.start_iou:
            why["start_not_detected"] += 1
        else:
            ok.append(c)
    print(f"[select] {len(ok)}/{len(cands)} candidates pass ({dict(why)})")
    # Frame check: where the mesh projects and where the detector finds the person agree only if
    # L, the metre scale and the lens are all right -- one number for the whole pipeline.
    g_iou = [c["goal_detect_iou"] for c in cands if "goal_detect_iou" in c]
    s_iou = [c["start_detect_iou"] for c in cands if "start_detect_iou" in c]
    iou_med = {"goal": float(np.median(g_iou)), "start": float(np.median(s_iou))}
    print(f"[select] median mesh-box vs detector IoU: goal {iou_med['goal']:.3f}, "
          f"start {iou_med['start']:.3f}")

    by_scene: dict[str, list] = defaultdict(list)
    for c in ok:
        by_scene[c["scene"]].append(c)
    episodes, short = [], {}
    for scene, cs in sorted(by_scene.items()):
        per_pl, paths, sec_n = Counter(), defaultdict(set), Counter()
        chosen, taken = [], set()
        for cap in range(args.per_placement, args.max_per_placement + 1):
            progress = True
            while progress and len(chosen) < args.per_scene:
                progress = False
                for sec in sorted(SECTORS, key=lambda s: sec_n[s]):
                    if len(chosen) >= args.per_scene:
                        break
                    # fewest-used placement first, so a top-up spreads over the scene's subjects
                    pool = sorted((c for c in cs if c["sector"] == sec and c["key"] not in taken
                                   and per_pl[c["placement"]] < cap
                                   and c["traj"] not in paths[c["placement"]]),
                                  key=lambda c: per_pl[c["placement"]])
                    if pool:
                        pick = pool[0]
                        chosen.append(pick)
                        taken.add(pick["key"])
                        per_pl[pick["placement"]] += 1
                        paths[pick["placement"]].add(pick["traj"])
                        sec_n[sec] += 1
                        progress = True
            if len(chosen) >= args.per_scene:
                break
        n_pl = len({c["placement"] for c in cs})
        if len(chosen) < args.per_scene:
            short[scene] = f"{len(chosen)}/{args.per_scene}"
        print(f"[select] {scene:12s} {len(chosen):3d} episodes from {n_pl} placements "
              f"(per placement {dict(Counter(per_pl.values()))}); sectors {dict(sec_n)}")
        episodes += chosen
    episodes.sort(key=lambda e: (e["placement"], e["key"]))
    ds = np.array([e["d_start"] for e in episodes])
    out = Path(args.out)
    out.write_text(json.dumps({
        "_what": "3DGS benchmark episodes (paper Table 5): real-world reconstructions with a placed "
                 "subject, the training sampler's camera paths, the training scorer's labels.",
        "_contract": "A MANIFEST: do not edit; a changed set is a new version.",
        "kind": "gs", "version": 1, "chunk_size": 8,
        "rules": {"goal_subject_lum": args.min_lum, "goal_detect_iou": args.goal_iou,
                  "start_detect_iou": args.start_iou, "per_placement": args.per_placement,
                  "per_scene": args.per_scene, "max_per_placement": args.max_per_placement},
        "dropped": dict(why), "under_target": short, "median_detect_iou": iou_med,
        "builder_report": report,
        "sector_mix": dict(Counter(e["sector"] for e in episodes)),
        "n_episodes": len(episodes), "mean_d_start": float(ds.mean()) if len(ds) else None,
        "episodes": episodes}, indent=1))
    print(f"[select] {len(episodes)} episodes -> {out}; d_start mean {ds.mean():.3f}; "
          f"under target {short or 'none'}")

    from PIL import Image, ImageDraw
    Wt, Ht, cols = 256, 192, 10
    for which in ("goal", "start"):
        for s in range(0, len(episodes), 60):
            page = episodes[s:s + 60]
            sheet = Image.new("RGB", (cols * Wt, -(-len(page) // cols) * Ht), "white")
            dr = ImageDraw.Draw(sheet)
            for j, e in enumerate(page):
                x, y = (j % cols) * Wt, (j // cols) * Ht
                sheet.paste(Image.open(e[f"{which}_image"]).convert("RGB").resize((Wt, Ht)), (x, y))
                dr.rectangle((x, y, x + Wt - 1, y + 14), fill=(0, 0, 0))
                dr.text((x + 3, y + 1), f"#{s + j} {e['scene'][:10]} {e['sector']} d{e['delta']}",
                        fill=(255, 255, 0))
            sheet.save(D / f"{which}_{s // 60:02d}.jpg", quality=85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
