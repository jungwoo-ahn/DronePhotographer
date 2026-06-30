"""Human-in-the-loop goal curation: build a contact-sheet of candidate goal frames,
then turn the picked ones into the eval goal set.

The eval goals are shot *profiles* (scene-agnostic 8-key V5 vectors). To make them
genuinely *good* shots (not just "well-composed random frames"), we surface REAL
rendered frames from held-out val scenes (read straight off disk — no Blender),
pre-filter to plausibly-decent compositions, sample some, and lay them out in a
labeled grid. You pick the good IDs; we extract those frames' profiles as goals.

Two modes:
  # 1) build the gallery -> gallery/gallery.png + gallery/candidates.json
  PYTHONPATH=. .venv/bin/python scripts/build_goal_gallery.py --n 96 --sample random

  # 2) turn your picks into the goal set
  PYTHONPATH=. .venv/bin/python scripts/build_goal_gallery.py \
      --pick "3,8,15,22,..." --candidates gallery/candidates.json --out-goals configs/eval/eval_goals.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.policy.common.annotations import iter_windows
from src.policy.common.goal_space import normalize_goal
from src.scoring.bbox_control import V5_SCORE_KEYS

REPO = Path(__file__).resolve().parents[1]


def read_val(path: str) -> set[str]:
    return {ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")}


def get_font(size: int):
    import matplotlib
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"),
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def is_good(s: dict, W: int, H: int, a) -> bool:
    """Loose pre-filter: in-frame, sensibly sized, not extreme angle. Human curates the rest."""
    if s["body_in_frame_ratio"] < a.min_bif:
        return False
    if not (a.occ_min <= s["occupancy"] <= a.occ_max):
        return False
    if abs(s["cam_to_obj_elevation_deg"]) > a.max_el:
        return False
    cx, cy = s["object_center_x"], s["object_center_y"]
    m = a.center_margin
    return m * W <= cx <= (1 - m) * W and m * H <= cy <= (1 - m) * H


def farthest_point(X: np.ndarray, n: int) -> list[int]:
    seed = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    chosen = [seed]
    d = np.linalg.norm(X - X[seed], axis=1)
    while len(chosen) < n and len(chosen) < len(X):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return chosen


def label(s: dict, W: int, H: int) -> tuple[str, str]:
    occ, el = s["occupancy"], s["cam_to_obj_elevation_deg"]
    cx, cy, az = s["object_center_x"], s["object_center_y"], s["cam_to_obj_azimuth_deg"]
    size = "closeup" if occ >= 40 else ("medium" if occ >= 22 else "wide")
    angle = "highangle" if el <= -12 else ("lowangle" if el >= 12 else "eyelevel")
    fx = cx / W
    horiz = "centered" if abs(fx - 0.5) < 0.12 else ("left" if fx < 0.5 else "right")
    fy = cy / H
    vert = "" if abs(fy - 0.5) < 0.15 else ("_top" if fy < 0.5 else "_low")
    oct_names = ["frontR", "right", "backR", "back", "backL", "left", "frontL", "front"]
    az_name = oct_names[int(((az + 22.5) % 360) // 45)]
    name = f"{size}_{horiz}{vert}_{angle}_{az_name}"
    desc = (f"{size} shot, {'centered' if horiz == 'centered' else horiz + '-third'}"
            f"{', high in frame' if vert == '_top' else ', low in frame' if vert == '_low' else ''}, "
            f"{angle} (el {el}°), azimuth {az}°, occupancy {occ}%")
    return name, desc


def stratified_sample(cand, n: int) -> list[int]:
    """Round-robin across (scene, azimuth-octant) buckets for max scene + angle diversity."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, c in enumerate(cand):
        scene = c[3].split("__")[0]
        az_oct = int(((c[1]["cam_to_obj_azimuth_deg"] + 22.5) % 360) // 45)
        buckets[(scene, az_oct)].append(i)
    rng = np.random.default_rng()
    for b in buckets.values():
        rng.shuffle(b)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    idx: list[int] = []
    while len(idx) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                idx.append(buckets[k].pop())
                if len(idx) >= n:
                    break
    return idx


def collect(a):
    val = read_val(a.val_names)
    cand = []  # (norm_vec, scores, image_path, placement, frame_idx)
    seen_imgs = set()
    for d in sorted(Path(a.data_root).iterdir()):
        if not d.is_dir():
            continue
        if a.scenes == "val" and d.name.split("__")[0] not in val:
            continue
        dj = d / "data.json"
        if not dj.exists():
            continue
        try:
            doc = json.loads(dj.read_text())
        except Exception:
            continue
        W, H = doc.get("render_width", 1024), doc.get("render_height", 768)
        for pair in (doc.get("render_records") or []):
            for rec in pair:
                s = rec.get("scores")
                if not s or not all(k in s for k in V5_SCORE_KEYS) or not is_good(s, W, H, a):
                    continue
                pr = rec.get("path_rel")
                if not pr:
                    continue
                img = str(d / pr)
                if not os.path.exists(img):
                    img = str(d / "renders" / pr)
                if img in seen_imgs or not os.path.exists(img):
                    continue
                seen_imgs.add(img)
                vec = normalize_goal(np.array([s[k] for k in V5_SCORE_KEYS], np.float32), V5_SCORE_KEYS)
                cand.append((vec, {k: int(s[k]) for k in V5_SCORE_KEYS}, img, d.name, rec.get("frame_idx")))
    return cand


def build_gallery(args):
    cand = collect(args)
    print(f"{len(cand)} candidate frames pass the (loose) composition filter in held-out scenes")
    if not cand:
        raise SystemExit("no candidates — loosen the filter")
    n = min(args.n, len(cand))
    if args.sample == "fps":
        X = np.stack([c[0] for c in cand])
        idx = farthest_point(X, n)
    elif args.sample == "stratified":
        idx = stratified_sample(cand, n)
    else:  # random — fresh entropy each run, so re-running re-rolls the sample
        idx = list(np.random.default_rng().choice(len(cand), size=n, replace=False))

    tw = args.thumb
    th = int(tw * 3 / 4)  # 1024x768 -> 4:3
    font_id, font_sm = get_font(22), get_font(13)
    tiles, meta = [], []
    for gid, i in enumerate(idx):
        _, sc, img, pl, fr = cand[i]
        im = Image.open(img).convert("RGB").resize((tw, th))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, tw, 26], fill=(0, 0, 0))
        dr.text((4, 1), f"#{gid}", fill=(255, 220, 0), font=font_id)
        dr.text((52, 6), f"occ{sc['occupancy']}  az{sc['cam_to_obj_azimuth_deg']}  el{sc['cam_to_obj_elevation_deg']}",
                fill=(170, 215, 255), font=font_sm)
        tiles.append(im)
        meta.append({"id": gid, "placement": pl, "frame_idx": fr, "image": img, "profile": sc})

    cols = args.cols
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * tw, rows * th), (24, 24, 28))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % cols) * tw, (k // cols) * th))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.save(out / "gallery.png")
    (out / "candidates.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'gallery.png'} ({grid.size[0]}x{grid.size[1]}) + candidates.json "
          f"with {len(meta)} tiles (#0..#{len(meta)-1}), sample={args.sample}")


def build_goals_from_picks(args):
    cands = {m["id"]: m for m in json.loads(Path(args.candidates).read_text())}
    picks = [int(x) for x in str(args.pick).replace(" ", "").split(",") if x != ""]
    goals, seen = [], {}
    for pid in picks:
        m = cands[pid]
        s = m["profile"]
        name, desc = label(s, 1024, 768)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        goals.append({"name": name, "description": desc, "profile": s,
                      "source_placement": m["placement"], "source_frame": m["frame_idx"],
                      "gallery_id": pid})
    out = Path(args.out_goals)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(goals, indent=2))
    print(f"wrote {len(goals)} curated goals -> {out}")
    for g in goals:
        p = g["profile"]
        print(f"  #{g['gallery_id']:>3} {g['name']:42s} occ{p['occupancy']:>3} az{p['cam_to_obj_azimuth_deg']:>3} "
              f"el{p['cam_to_obj_elevation_deg']:>4} cx{p['object_center_x']:>4} cy{p['object_center_y']:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=250)
    ap.add_argument("--sample", choices=["stratified", "random", "fps"], default="stratified",
                    help="stratified = balanced across scenes+angles (default); random; fps=edge cases")
    ap.add_argument("--scenes", choices=["all", "val"], default="all",
                    help="all = example frames from every scene (goals are scene-agnostic); val = held-out only")
    ap.add_argument("--val-names", default="configs/policy/val_scenes.txt")
    ap.add_argument("--data-root", default="data/trajectories")
    ap.add_argument("--out", default="gallery")
    # loose composition pre-filter (human curates from here)
    ap.add_argument("--min-bif", type=int, default=85)
    ap.add_argument("--occ-min", type=int, default=12)
    ap.add_argument("--occ-max", type=int, default=65)
    ap.add_argument("--max-el", type=int, default=42)
    ap.add_argument("--center-margin", type=float, default=0.06)
    # pick mode
    ap.add_argument("--pick", default=None, help="comma-separated gallery IDs -> goals")
    ap.add_argument("--candidates", default="gallery/candidates.json")
    ap.add_argument("--out-goals", default="configs/eval/eval_goals.json")
    args = ap.parse_args()

    if args.pick is not None:
        build_goals_from_picks(args)
    else:
        build_gallery(args)


if __name__ == "__main__":
    main()
