"""Audit scene scales + positions using gemini-3-flash-preview as a reviewer.

For each scene, picks N=5 placements from N different objects (prefers standing
humans, falls back to other accepted placements), calls the full placement VLM
(quality + position + scale) with gemini-3, drops views where
surroundings_empty=true (no reference visible), aggregates the rest by median.

Flags a scene if median scale_factor is far from 1.0, OR if median quality is
low, OR if too few valid views remained after filtering.

Writes _scale_audit.json into the run dir. Does not modify pair JSONs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.vlm.api import call_vlm

STANDING_HUMAN_KEYWORDS = (
    "andrew", "charles", "john", "koky", "luke", "wenceslavus", "utel",
    "oscar", "moma", "business", "farmer", "cute",
)
SITTING_KEYWORDS = ("sitting", "sit", "crouch", "squat", "kneel", "stool")


def pick_n_diverse_placements(pair_jsons: list[Path], n: int = 5) -> list[tuple]:
    """Return up to n (pair_path, placement, object_name) tuples, each from a
    different object pair. Prefers standing-human objects; skips sitting ones."""
    standing = []
    other = []
    for pair_path in pair_jsons:
        try:
            d = json.load(open(pair_path))
        except Exception:
            continue
        placements = d.get("placements", [])
        if not placements:
            continue
        stem = pair_path.stem
        if "__" not in stem:
            continue
        obj_name = stem.split("__", 1)[1]
        obj_lc = obj_name.lower()
        if any(k in obj_lc for k in SITTING_KEYWORDS):
            continue
        for pl in placements:
            if not pl.get("accepted"):
                continue
            img = (pl.get("attempts") or [{}])[0].get("preview_image")
            if not img or not os.path.exists(img):
                continue
            entry = (pair_path, pl, obj_name)
            if any(k in obj_lc for k in STANDING_HUMAN_KEYWORDS):
                standing.append(entry)
            else:
                other.append(entry)
            break  # take 1 per pair
    return (standing + other)[:n]


def audit_placement(pair_path: Path, placement: dict, object_name: str,
                    scene_name: str, vlm_config: dict) -> dict:
    """Call full placement VLM on one image; return parsed VLM dict or error."""
    img = placement["attempts"][0]["preview_image"]
    try:
        result = call_vlm(
            image_path=img,
            object_name=object_name,
            scene_name=scene_name,
            camera_radius=5.0,
            camera_elevation=15.0,
            vlm_config=vlm_config,
        )
    except Exception as e:
        return {"object": object_name, "image": img, "error": str(e)}
    if not result:
        return {"object": object_name, "image": img, "error": "vlm returned None"}
    return {
        "object": object_name,
        "image": img,
        "quality": result.get("quality"),
        "surroundings_empty": result.get("surroundings_empty", False),
        "scale_factor": (result.get("scale_factor") or 1.0),
        "reasoning": (result.get("reasoning") or "")[:200],
    }


def audit_scene(scene: str, pair_jsons: list[Path], vlm_config: dict, n: int = 5) -> dict:
    picks = pick_n_diverse_placements(pair_jsons, n)
    if not picks:
        return {"scene": scene, "status": "no_reference", "flagged": False}
    saved_scale = None
    samples = []
    for pair_path, pl, obj in picks:
        if saved_scale is None:
            try:
                saved_scale = json.load(open(pair_path)).get("scene_scale", 1.0)
            except Exception:
                saved_scale = 1.0
        samples.append(audit_placement(pair_path, pl, obj, scene, vlm_config))

    # Aggregate, dropping surroundings_empty + errors
    valid = [s for s in samples
             if "error" not in s
             and not s.get("surroundings_empty")
             and s.get("quality") is not None]
    n_valid = len(valid)
    if n_valid < 2:
        return {
            "scene": scene,
            "status": "insufficient_signal",
            "samples_total": len(samples),
            "samples_valid": n_valid,
            "samples": samples,
            "flagged": True,  # not enough signal = worth flagging for review
            "reason_for_flag": "fewer than 2 views had visible scene reference",
        }

    qualities = [float(s["quality"]) for s in valid]
    sfs = [float(s["scale_factor"]) for s in valid]
    med_q = median(qualities)
    med_sf = median(sfs)

    flagged = abs(med_sf - 1.0) > 0.3 or med_q < 6.0
    flag_reasons = []
    if abs(med_sf - 1.0) > 0.3:
        flag_reasons.append(f"median_sf={med_sf:.2f}")
    if med_q < 6.0:
        flag_reasons.append(f"median_q={med_q:.1f}")

    return {
        "scene": scene,
        "status": "ok",
        "saved_scene_scale": saved_scale,
        "samples_total": len(samples),
        "samples_valid": n_valid,
        "median_scale_factor": med_sf,
        "median_quality": med_q,
        "flagged": flagged,
        "flag_reasons": flag_reasons,
        "samples": [
            {"object": s["object"][:50],
             "q": s.get("quality"),
             "sf": s.get("scale_factor"),
             "empty": s.get("surroundings_empty"),
             "reason": s.get("reasoning", "")[:80],
             "error": s.get("error")}
            for s in samples
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--config", default="configs/vlm_placement.json")
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--n_per_scene", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default=None)
    p.add_argument("--scenes", nargs="*", default=None, help="limit to these scene prefixes")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config = json.load(open(args.config))
    vlm_config = dict(config["vlm"])
    vlm_config["model"] = args.model
    out_path = Path(args.output) if args.output else run_dir / "_scale_audit.json"

    by_scene: dict[str, list[Path]] = {}
    for f in os.listdir(run_dir):
        if not f.endswith(".json") or "__" not in f or f.startswith("_"):
            continue
        scene = f.split("__", 1)[0]
        by_scene.setdefault(scene, []).append(run_dir / f)

    if args.scenes:
        by_scene = {k: v for k, v in by_scene.items() if k in args.scenes}

    print(f"Auditing {len(by_scene)} scenes × {args.n_per_scene} placements each "
          f"with model={args.model} (workers={args.workers})")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(audit_scene, s, jsons, vlm_config, args.n_per_scene): s
                   for s, jsons in by_scene.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = "FLAG" if r.get("flagged") else r["status"]
            extra = ""
            if "median_scale_factor" in r:
                extra = f"  med_sf={r['median_scale_factor']:.2f} med_q={r['median_quality']:.1f}  ({r['samples_valid']}/{r['samples_total']})"
            print(f"[{i:3d}/{len(by_scene)}] {tag:<22} {r['scene'][:50]}{extra}")

    results.sort(key=lambda r: (not r.get("flagged"), r["status"] == "ok", r["scene"]))
    out = {
        "model": args.model,
        "n_per_scene": args.n_per_scene,
        "total_scenes": len(by_scene),
        "flagged_count": sum(1 for r in results if r.get("flagged")),
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    elapsed = time.time() - t0
    print(f"\nWrote {out_path}  ({elapsed:.1f}s, {out['flagged_count']} flagged)")


if __name__ == "__main__":
    main()
