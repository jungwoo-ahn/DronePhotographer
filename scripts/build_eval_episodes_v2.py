"""The shared closed-loop episode set, version 2: every goal is a photograph of the subject.

Version 1 (`configs/eval_episodes_v1.json`, `scripts/build_eval_episodes.py`) made a policy
comparison possible at all, and it is still the record behind every number published before
2026-10-08. Four things keep it from carrying the paper's tables:
  * about 30% of its goals show no person: a wall, darkness, or a torso with the head cut off;
    the stage-3 profile is computed from the projected mesh and cannot see any of that
  * 15-18% of starts (measured on the reference re-shoot) show a wall instead of the subject
  * two of its eight scenes cannot be rendered the way the training frames were: Parking_70ae1f27
    segfaults Cycles, and Modern-Warehouse stays ~31/255 off under the v7 render profile
  * 54 usable episodes give a success-rate interval of about +-13 points, wider than most of the
    differences the paper needs to resolve

Version 2 keeps v1's window pool (chunk 8, at most two windows per trajectory), so
`closed_loop_eval.py --episodes-from` replays it unchanged, and adds rules, with no human pass:
  * scenes  the held-out scenes minus the two above
  * goal    head in frame; the subject's box not dark; a person detected where the mesh projects
  * start   a person detected where the mesh projects (a looser overlap: starts are often partial)
  * mix     --per-scene episodes per scene, dealt round-robin over the eight view sectors,
            alternating short (< 16 frames) and long start distances, one episode per placement
            and a second one (from another trajectory) only when a scene runs out

    sbatch --qos=own scripts/sbatch_build_eval_v2.sh          # the detector pass wants a GPU
venv: .venv-analysis (ultralytics + headless OpenCV).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)

import numpy as np  # noqa: E402

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import (DEFAULT_GOAL_KEYS, RENDER_HEIGHT,  # noqa: E402
                                   RENDER_WIDTH, SUBJECT_BEARING_KEY, goal_vector)
from src.common.visibility import best_iou, clip_box, detect_people, subject_luminance  # noqa: E402

SECTOR_ORDER = ("front", "front-right", "right", "back-right",
                "back", "back-left", "left", "front-left")
BINS = ("short", "long")
CHUNK_SIZE, MAX_PER_PAIR = 8, 2          # v1's pool; the manifest replay looks windows up in it

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=f"{V12}/data/trajectories/v7_stage2_renders_lookat075")
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--exclude-scenes", nargs="*",
                default=["Parking_70ae1f27", "Modern-Warehouse_a5eea495"])
ap.add_argument("--per-scene", type=int, default=40)
ap.add_argument("--long-from", type=int, default=16, help="start distance (frames) that counts as long")
ap.add_argument("--min-lum", type=float, default=20.0,
                help="mean luminance of the goal's subject box, 0-255; drops black frames, keeps "
                     "the dim indoor scenes")
ap.add_argument("--goal-iou", type=float, default=0.5)
ap.add_argument("--start-iou", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=f"{V12}/configs/eval_episodes_v2.json")
ap.add_argument("--sheet-dir", default=f"{V12}/runs/bench_v2")
args = ap.parse_args()


def _v1():
    spec = importlib.util.spec_from_file_location(
        "build_eval_episodes", Path(V12) / "scripts" / "build_eval_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidates(why: Counter) -> list[dict]:
    """Every pool window that passes the score-only rules, with its clipped boxes."""
    keep = set(json.loads(Path(args.val_scenes).read_text())["scenes"]) - set(args.exclude_scenes)
    root, out = Path(args.root), []
    for name in sorted(d for d in os.listdir(root) if d.split("__")[0] in keep):
        obj = name.split("__", 1)[1] if "__" in name else name
        path = root / name / "data.json"
        if obj in DEFAULT_EXCLUDE_OBJECTS or not path.exists():
            continue
        try:
            windows = list(iter_goal_start_windows(path, chunk_size=CHUNK_SIZE,
                                                   max_per_pair=MAX_PER_PAIR))
        except Exception:  # noqa: BLE001
            why["unreadable"] += 1
            continue
        for w in windows:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                why["goal_vector"] += 1
                continue
            if w.goal_frame.raw.get("head_in_frame") != 1:
                why["head_out_of_frame"] += 1
                continue
            gb = clip_box(w.goal_frame.raw.get("bbox_xyxy_full") or (0, 0, 0, 0), RENDER_WIDTH, RENDER_HEIGHT)
            sb = clip_box(w.start.raw.get("bbox_xyxy_full") or (0, 0, 0, 0), RENDER_WIDTH, RENDER_HEIGHT)
            if gb is None or sb is None:
                why["box_off_frame"] += 1
                continue
            if not (Path(w.goal_frame.image).exists() and Path(w.start.image).exists()):
                why["missing_image"] += 1
                continue
            bearing = float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
            delta = abs(w.goal_frame.frame_idx - w.start_frame_idx)
            out.append({"name": name, "scene": name.split("__")[0], "path": path, "w": w, "g": g,
                        "bearing": bearing, "sector": sector8(bearing), "delta": delta,
                        "bin": "long" if delta >= args.long_from else "short",
                        "pair": w.pair_idx, "goal_box": gb, "start_box": sb,
                        "key": f"{name}:p{w.pair_idx:02d}:s{w.start_frame_idx:02d}"
                               f":g{w.goal_frame.frame_idx:02d}"})
    return out


def select(cands: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """Deal `quota` episodes round-robin over the sectors, balancing short/long within each."""
    cells: dict[tuple, list] = defaultdict(list)
    for c in cands:
        cells[(c["sector"], c["bin"])].append(c)
    for v in cells.values():
        rng.shuffle(v)
    chosen, count, taken = [], Counter(), set()
    pairs_used: dict[str, set] = defaultdict(set)
    for cap in (1, 2):                    # a second episode per placement only if the scene runs out
        progress = True
        while len(chosen) < quota and progress:
            progress = False
            for sec in SECTOR_ORDER:
                if len(chosen) >= quota:
                    break
                for b in sorted(BINS, key=lambda x: (count[(sec, x)], BINS.index(x))):
                    pick = next((c for c in cells[(sec, b)] if c["key"] not in taken
                                 and len(pairs_used[c["name"]]) < cap
                                 and c["pair"] not in pairs_used[c["name"]]), None)
                    if pick is not None:
                        chosen.append(pick)
                        taken.add(pick["key"])
                        pairs_used[pick["name"]].add(pick["pair"])
                        count[(sec, b)] += 1
                        progress = True
                        break
    return chosen


def sheets(eps: list[dict], which: str) -> list[str]:
    from PIL import Image, ImageDraw

    out_dir = Path(args.sheet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, cols, per = 256, 192, 10, 60
    paths = []
    for s in range(0, len(eps), per):
        page = eps[s:s + per]
        sheet = Image.new("RGB", (cols * W, -(-len(page) // cols) * H), "white")
        d = ImageDraw.Draw(sheet)
        for j, e in enumerate(page):
            x, y = (j % cols) * W, (j // cols) * H
            sheet.paste(Image.open(e[f"{which}_image"]).convert("RGB").resize((W, H)), (x, y))
            d.rectangle((x, y, x + W - 1, y + 14), fill=(0, 0, 0))
            d.text((x + 3, y + 1), f"#{s + j} {e['scene'][:12]} {e['sector']} d{e['delta']}",
                   fill=(255, 255, 0))
        p = out_dir / f"{which}_{s // per:02d}.jpg"
        sheet.save(p, quality=85)
        paths.append(str(p))
    return paths


def main() -> int:
    t0 = time.time()
    why: Counter = Counter()
    cands = candidates(why)
    print(f"[v2] {len(cands)} windows pass the score rules (dropped {dict(why)}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    lum = {}
    for c in cands:
        f = c["w"].goal_frame.image
        if f not in lum:
            lum[f] = subject_luminance(f, c["goal_box"])
    lit = [c for c in cands if lum[c["w"].goal_frame.image] >= args.min_lum]
    print(f"[v2] light keeps {len(lit)}/{len(cands)} [{time.time()-t0:.0f}s]", flush=True)

    det = detect_people([c["w"].goal_frame.image for c in lit] + [c["w"].start.image for c in lit])
    seen = []
    for c in lit:
        c["goal_iou"] = best_iou(det[c["w"].goal_frame.image], c["goal_box"])
        c["start_iou"] = best_iou(det[c["w"].start.image], c["start_box"])
        if c["goal_iou"] < args.goal_iou:
            why["goal_not_detected"] += 1
        elif c["start_iou"] < args.start_iou:
            why["start_not_detected"] += 1
        else:
            seen.append(c)
    print(f"[v2] detector keeps {len(seen)}/{len(lit)} [{time.time()-t0:.0f}s]", flush=True)

    by_scene: dict[str, list] = defaultdict(list)
    for c in seen:
        by_scene[c["scene"]].append(c)
    rng = random.Random(args.seed)
    v1 = _v1()
    episodes, report = [], {}
    for scene in sorted(by_scene):
        chosen = select(by_scene[scene], args.per_scene, rng)
        mix = Counter((c["sector"], c["bin"]) for c in chosen)
        report[scene] = {"available": len(by_scene[scene]), "chosen": len(chosen),
                         "placements": len({c["name"] for c in chosen}),
                         "sectors": {s: mix[(s, "short")] + mix[(s, "long")] for s in SECTOR_ORDER},
                         "short": sum(mix[(s, "short")] for s in SECTOR_ORDER)}
        for c in chosen:
            e = v1.materialise(c["name"], c["path"], c["w"], c["g"], c["bearing"], c["sector"])
            e.update({"delta_bin": c["bin"], "goal_detect_iou": round(c["goal_iou"], 3),
                      "start_detect_iou": round(c["start_iou"], 3),
                      "goal_subject_lum": round(lum[c["w"].goal_frame.image], 1)})
            episodes.append(e)
        print(f"[v2] {scene[:40]:40s} {len(chosen):3d}/{args.per_scene} from "
              f"{report[scene]['available']} ({report[scene]['placements']} placements)  "
              f"sectors {report[scene]['sectors']}  short {report[scene]['short']}", flush=True)

    ds = np.array([e["d_start"] for e in episodes])
    sector_mix = Counter(e["sector"] for e in episodes)
    short = {s: report[s]["chosen"] for s in report if report[s]["chosen"] < args.per_scene}
    doc = {
        "_what": "The shared closed-loop episode set, version 2: held-out scenes that render like "
                 "the training data, goals that are photographs of the subject, starts that show it. "
                 "Every policy in one table runs exactly these.",
        "_contract": "A MANIFEST, not a rule. Do not edit; a changed set is a new version. Numbers "
                     "on v1 and v2 are not comparable.",
        "_verify": "closed_loop_eval.py --episodes-from recomputes d_start for every episode and "
                   "refuses to run if it differs by more than 1e-6.",
        "version": 2,
        "source_root": args.root, "val_scenes": args.val_scenes,
        "excluded_scenes": args.exclude_scenes,
        "scenes": sorted(report),
        "chunk_size": CHUNK_SIZE, "max_per_pair": MAX_PER_PAIR, "seed": args.seed,
        "rules": {"goal_head_in_frame": True, "min_goal_subject_lum": args.min_lum,
                  "goal_detect_iou": args.goal_iou, "start_detect_iou": args.start_iou,
                  "per_scene": args.per_scene, "long_from_frames": args.long_from,
                  "detector": "yolo11l-pose, person class, conf >= 0.25"},
        "dropped": dict(why), "scene_report": report, "under_target": short,
        "sector_mix": {s: sector_mix[s] for s in SECTOR_ORDER},
        "n_episodes": len(episodes), "mean_d_start": float(ds.mean()),
        "episodes": episodes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"[v2] {len(episodes)} episodes, d_start mean {ds.mean():.3f} median {np.median(ds):.3f}; "
          f"sectors {doc['sector_mix']}; under target {short or 'none'}")
    print(f"[v2] wrote {out}; sheets {sheets(episodes, 'goal') + sheets(episodes, 'start')} "
          f"[{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
