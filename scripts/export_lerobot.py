"""Export `goal_start` samples into a LeRobot dataset for pretrained-VLA fine-tuning.

Produces the (image, language-goal, action-chunk) triples our custom baselines train on,
but in the canonical LeRobot format that pretrained **pi0.5** (LeRobot `pi05`) and
**GR00T N1.5** consume directly. Written via the official `LeRobotDataset` API (not a
hand-rolled writer) so stats / episode meta / video encoding are exactly what those
loaders expect.

    conda activate vla
    PYTHONPATH=. python scripts/export_lerobot.py \
        --out /home/nas5/jooyeolyun/datasets/drone_data/lerobot_pi05_v1 \
        --max-episodes 40000

Mapping (one EPISODE per (start, goal) sample, matching our window semantics):
  * observation.images.image : the start->goal walk frames (chunk_size + 1), resized square
  * observation.state        : ZEROS (no proprio) — our task is image + language goal only,
                               matching the DP / custom-VLA baselines. Train pi05 with
                               STATE normalization = IDENTITY (zero-variance would break
                               the default q01/q99 normalization).
  * action                   : 10D = [dx,dy,dz, rot6d(6), shoot]  (already produced 10-wide
                               by `_compute_action_chunk`; last frame padded with zeros)
  * task                     : the natural-language shot-profile goal prompt (goal conditioning)

Scene-level held-out split (V12 `val_scenes.json`): placements are partitioned by scene
BEFORE the sector-balanced fill so each side is balanced on its own — filtering a globally
balanced draw afterwards would re-skew both sides.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.policy.common.annotations import iter_goal_start_windows
from src.policy.common.dataset_base import _compute_action_chunk
from src.policy.common.facing import sector8
from src.policy.common.goal_space import goal_vector
from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt

SUBJECT_BEARING_KEY = "subject_bearing_deg"
SECTOR_ORDER = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)
VIDEO_KEY = "observation.images.image"


def _load_square(path: Path, size: int) -> np.ndarray:
    """RGB uint8 HWC, resized to `size`x`size` (paligemma / Eagle-2 want square)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_episodes(args):
    """Assemble (frames, actions[10], prompt, scene, sector) episodes, scene-split + balanced.

    Returns (train_eps, val_eps) where each ep is a dict. This layer is format-independent;
    the LeRobot write happens in `write_dataset`.
    """
    placements: list[tuple[str, Path]] = []
    for root in args.roots:
        r = Path(root)
        if not r.is_dir():
            print(f"  skip missing root {r}")
            continue
        for d in sorted(os.listdir(r)):
            if (r / d / "data.json").exists():
                placements.append((d, r / d / "data.json"))
    random.seed(args.seed)
    random.shuffle(placements)
    print(f"placements: {len(placements)} across {len(args.roots)} root(s)", flush=True)

    val_scenes: frozenset[str] = frozenset()
    if args.val_scenes:
        val_scenes = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
        print(f"val scenes: {len(val_scenes)} held out whole (from {args.val_scenes})")
    split_of = {name: ("val" if name.split("__")[0] in val_scenes else "train")
                for name, _ in placements}
    n_val_pl = sum(1 for v in split_of.values() if v == "val")
    print(f"  train placements {len(placements)-n_val_pl}  |  val placements {n_val_pl}")

    exclude = set(args.exclude or [])
    bearing_idx = list(NL_GOAL_KEYS).index(SUBJECT_BEARING_KEY)
    t0 = time.time()
    near_count = 0

    def fill(split: str, budget: int):
        nonlocal near_count
        out, sectors, objects = [], Counter(), Counter()
        cap = max(1, budget // 8) if args.balance_sectors else budget
        mine = [(n, p) for n, p in placements if split_of[n] == split]

        def _full() -> bool:
            if not args.balance_sectors:
                return len(out) >= budget
            return all(sectors[s] >= cap for s in SECTOR_ORDER)

        for i, (name, path) in enumerate(mine):
            if _full() or len(out) >= budget:
                break
            obj = name.split("__", 1)[1] if "__" in name else name
            if obj in exclude:
                continue
            try:
                windows = list(iter_goal_start_windows(
                    path, chunk_size=args.chunk_size, max_per_pair=args.per_placement))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {name[:40]}: {exc}")
                continue
            random.shuffle(windows)
            taken = 0
            for w in windows:
                if taken >= args.per_placement or len(out) >= budget:
                    break
                g = goal_vector(w.goal_frame.raw, NL_GOAL_KEYS)
                if not np.isfinite(g).all():
                    continue
                sec = sector8(float(g[bearing_idx]))
                if args.balance_sectors and sectors[sec] >= cap:
                    continue
                frames = [Path(k.image) for k in w.keyframes]
                if not all(f.exists() for f in frames):
                    continue
                pose = _compute_action_chunk(w)              # (chunk, 10) incl. shoot
                out.append({
                    "frames": frames,
                    "actions": pose.astype(np.float32),
                    "prompt": goal_prompt(g, NL_GOAL_KEYS, crop=w.goal_frame.raw),
                    "scene": name.split("__")[0],
                    "sector": sec,
                })
                sectors[sec] += 1
                if abs(w.goal_frame.frame_idx - w.start_frame_idx) < args.chunk_size:
                    near_count += 1
                objects[obj] += 1
                taken += 1
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(mine)} placements, {len(out)} eps, "
                      f"{time.time()-t0:.0f}s", flush=True)

        tot = sum(sectors.values()) or 1
        print(f"[{split}] episodes {len(out)}  objects {len(objects)}  ({time.time()-t0:.0f}s)")
        print(f"[{split}] sector mix: " +
              "  ".join(f"{s}={100*sectors[s]/tot:.0f}%" for s in SECTOR_ORDER))
        if args.balance_sectors:
            short = {s: sectors[s] for s in SECTOR_ORDER if sectors[s] < cap}
            print(f"[{split}] target {cap}/sector: " +
                  ("all filled" if not short else f"UNDER {short}"))
        return out

    n_val = int(args.max_episodes * args.val_fraction) if val_scenes else 0
    val_eps = fill("val", n_val) if n_val else []
    train_eps = fill("train", args.max_episodes - len(val_eps))
    tr_sc = {e["scene"] for e in train_eps}
    va_sc = {e["scene"] for e in val_eps}
    assert not (tr_sc & va_sc), f"scene leaked: {sorted(tr_sc & va_sc)[:5]}"
    total = len(train_eps) + len(val_eps)
    print(f"\ntotal {total}  (train {len(train_eps)} / {len(tr_sc)} scenes, "
          f"val {len(val_eps)} / {len(va_sc)} scenes, overlap 0)")
    print(f"near-goal (delta<{args.chunk_size}): {near_count}/{total} = "
          f"{100*near_count/max(1,total):.0f}%")
    return train_eps, val_eps


def _write_one(episodes, out_path, repo_id, args):
    """Write ONE split's episodes as a standalone LeRobot dataset (official API)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out_path)
    if out.exists():
        import shutil
        shutil.rmtree(out)

    state_dim = args.state_dim
    img_dtype = "video" if args.video else "image"
    features = {
        VIDEO_KEY: {"dtype": img_dtype, "shape": (args.resize, args.resize, 3),
                    "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (10,),
                   "names": ["dx", "dy", "dz", "r0", "r1", "r2", "r3", "r4", "r5", "shoot"]},
    }
    # observation.state omitted when state_dim==0 (GR00T: a degenerate all-zeros proprio
    # channel risks div-by-zero in GR00T's internal normalization, so drop it entirely —
    # the task is image + language goal only). pi0.5 keeps a zero state (STATE=IDENTITY).
    if state_dim > 0:
        features["observation.state"] = {"dtype": "float32", "shape": (state_dim,),
                                         "names": [f"s{i}" for i in range(state_dim)]}
    create_kw = dict(
        repo_id=repo_id, fps=args.fps, root=str(out),
        robot_type="blender_camera", features=features, use_videos=bool(args.video),
    )
    if args.video:
        # Video path (kept as an option). For our TINY episodes it is pathologically slow:
        # batch_encoding_size>1 is broken in 0.6.2 (_meta.episodes None during the batched
        # encode) and per-episode/streaming encode has O(n)-per-episode ffmpeg overhead.
        # Prefer image storage (--video 0) for this data shape.
        if args.streaming:
            create_kw["streaming_encoding"] = True
        else:
            create_kw["batch_encoding_size"] = args.batch_encoding
            create_kw["metadata_buffer_size"] = args.metadata_buffer
    else:
        # Image dataset: one PNG per frame, parallel writers, no encoding. Linear + robust
        # for many-tiny-episodes; pi05/GR00T read observation.images.* the same either way.
        create_kw["image_writer_processes"] = args.image_writer_processes
        create_kw["image_writer_threads"] = args.image_writer_threads
    ds = LeRobotDataset.create(**create_kw)

    zero_state = np.zeros(state_dim, dtype=np.float32) if state_dim > 0 else None
    t0 = time.time()
    for ei, ep in enumerate(episodes):
        acts = ep["actions"]                          # (chunk, 10)
        # Emit ONLY frames with a real outgoing action (len(acts)); skip the episode's final
        # frame, which has no action. It used to be padded with a zero action, but those ~11%
        # zero rows contaminated the action normalization stats: rot6d r0/r4 (real range
        # [0.994, 1.0]) had their quantiles dragged to [0, 1] by the zeros, crushing the real
        # rotation signal into <1% of the normalized range and crippling rotation learning
        # (esp. pi0.5 under QUANTILES norm). The dropped frame was never a valid training
        # sample (no target action), so this only removes contamination.
        for f in range(len(acts)):
            act = acts[f]
            frame = {                                 # task is a frame key, not a kwarg
                VIDEO_KEY: _load_square(ep["frames"][f], args.resize),
                "action": act.astype(np.float32),
                "task": ep["prompt"],
            }
            if zero_state is not None:
                frame["observation.state"] = zero_state
            ds.add_frame(frame)
        ds.save_episode()
        if (ei + 1) % 500 == 0:
            print(f"  [{out.name}] wrote {ei+1}/{len(episodes)} episodes  {time.time()-t0:.0f}s",
                  flush=True)

    # finalize(): required on lerobot >=0.6 to flush buffered episode/video metadata (else
    # load fails with KeyError 'videos/observation.images.image/chunk_index').
    print(f"  [{out.name}] finalizing...", flush=True)
    ds.finalize()
    # Scene provenance for THIS split (LeRobot episode meta doesn't carry our scene name).
    (out / "meta" / "drone_split.json").write_text(json.dumps(
        {"scenes": [e["scene"] for e in episodes], "n": len(episodes)}, indent=1))
    print(f"wrote {out}: {len(episodes)} episodes ({time.time()-t0:.0f}s)", flush=True)


def write_dataset(train_eps, val_eps, args):
    """Write train and val as SEPARATE datasets (<out> and <out>_val), so lerobot-train can
    train on the train split alone and the held-out val scenes stay out of training — the
    fair-comparison requirement shared with the DP / Cosmos baselines."""
    _write_one(train_eps, args.out, args.repo_id, args)
    if val_eps:
        _write_one(val_eps, f"{args.out}_val", f"{args.repo_id}_val", args)
        print(f"\nTRAIN -> {args.out} ({len(train_eps)} eps)  |  "
              f"VAL -> {args.out}_val ({len(val_eps)} eps)  | scene-disjoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/trajectories_full"])
    ap.add_argument("--out", required=True, help="dataset root dir")
    ap.add_argument("--repo_id", default="drone/lerobot_pi05_v1")
    ap.add_argument("--max-episodes", type=int, default=40000, dest="max_episodes")
    ap.add_argument("--per-placement", type=int, default=4, dest="per_placement")
    ap.add_argument("--chunk-size", type=int, default=8, dest="chunk_size")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--state-dim", type=int, default=9, dest="state_dim")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--balance-sectors", type=int, default=1, dest="balance_sectors")
    ap.add_argument("--val-scenes", default="configs/policy/val_scenes.json", dest="val_scenes")
    ap.add_argument("--val-fraction", type=float, default=0.10, dest="val_fraction")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--batch-encoding", type=int, default=1000, dest="batch_encoding",
                    help="episodes per video-encode batch (avoids O(n^2) re-mux)")
    ap.add_argument("--metadata-buffer", type=int, default=1000, dest="metadata_buffer",
                    help="episodes per metadata parquet flush")
    ap.add_argument("--video", type=int, default=0,
                    help="1=video(mp4) storage, 0=image(png) storage (default; fast for tiny episodes)")
    ap.add_argument("--streaming", type=int, default=1,
                    help="[video only] use streaming_encoding (1) vs batch/synchronous (0)")
    ap.add_argument("--image-writer-processes", type=int, default=0, dest="image_writer_processes")
    ap.add_argument("--image-writer-threads", type=int, default=8, dest="image_writer_threads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="assemble episodes but don't write")
    args = ap.parse_args()

    train_eps, val_eps = build_episodes(args)
    if args.dry_run:
        print("\n[dry-run] not writing dataset")
        if train_eps:
            e = train_eps[0]
            print(f"sample: {len(e['frames'])} frames, action {e['actions'].shape}, "
                  f"scene={e['scene']}, sector={e['sector']}")
            print(f"prompt: {e['prompt']}")
        return
    write_dataset(train_eps, val_eps, args)


if __name__ == "__main__":
    main()
