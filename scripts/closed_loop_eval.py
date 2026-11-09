"""Closed-loop evaluation: does the policy actually REACH the requested shot?

Training curves and the goal-dependence probe both say the policy *uses* the goal.
Neither says it *gets there*. v11 passed the first test and failed this one — its
world-model variant looked better on sampled action MSE while being worse in
rollout — so this is the measurement that decides anything.

Protocol, per episode:
  1. Pick a real (start, goal) pair from a held-out placement.
  2. Put the camera at the start pose in Blender and render what it sees.
  3. Ask the policy for an 8-step action chunk, conditioned on that frame and on
     the goal prompt.
  4. Execute the chunk in the Blender env, re-rendering as it goes.
  5. Measure how much closer to the goal pose the camera ended up.

The headline number is improvement over doing nothing:

    improvement = d(start, goal) − d(end, goal)

positive means the camera moved toward the requested shot. Reported against the
no-op baseline explicitly, because a policy that barely moves scores ~0 and a
policy that thrashes scores negative — both are failures that a raw distance
would flatter. `d` is the geometric pose distance from `src/common/reward.py`
(view angle, apparent size and aim, all in radians), the same metric the value
target uses.

Usage (see scripts/sbatch_closed_loop_eval.sh for the wrapper that sets the env):

    python scripts/closed_loop_eval.py \
        --checkpoint runs/train/.../checkpoints/iter_000005000 \
        --episodes 24 --out runs/closed_loop/iter5000.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)
# Code comes from the checkout this script lives in (a worktree, say); data paths stay on V12.
# From the main checkout the two are the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Run from the framework root: it resolves several assets by RELATIVE path
# (e.g. model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json), so
# chdir-ing anywhere else makes model construction fail with FileNotFoundError.
CF_ROOT = f"{V12}/repos/cosmos-framework"
sys.path.insert(0, CF_ROOT)
os.chdir(CF_ROOT)

import numpy as np

# Imported ahead of argparse so `--val-scenes` can default to it. It is ABSOLUTE: this
# script does `os.chdir(CF_ROOT)` (Cosmos resolves model configs by relative path), so a
# relative default is looked for inside the vendored checkout. That already killed a
# training run once; re-introducing it here as a new flag killed all three rollouts.
# Mirrors src/data/cosmos_camera_dataset.DEFAULT_VAL_SCENES. Not imported from there: that
# module pulls cosmos_framework in at import time, and this script now also runs in the
# LeRobot venv for the baselines. tests/test_policy_adapters.py pins the two together.
DEFAULT_VAL_SCENES = str(Path(V12) / "configs" / "val_scenes.json")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--episodes", type=int, default=24)
ap.add_argument("--held-out-only", type=int, default=1,
                help="skip placements the export actually consumed (1 = on). NOT optional for "
                     "a generalization claim: the export takes 4 episodes from each of only "
                     "~1000 placements, and this script used to draw from the same shuffled "
                     "list, so every evaluated placement was one the model trained on.")
ap.add_argument("--export-seed", type=int, default=0, help="must match the export")
ap.add_argument("--export-max-episodes", type=int, default=4000, help="must match the export")
ap.add_argument("--export-per-placement", type=int, default=4, help="must match the export")
ap.add_argument("--per-sector", type=int, default=0,
                help="cap episodes per view sector (0 = off). Balances coverage across the "
                     "eight sectors instead of following the back-heavy data, so raising "
                     "--episodes buys variety rather than more back views.")
ap.add_argument("--chunks", type=int, default=0,
                help="action chunks per episode. 0 = ADAPTIVE: ceil(delta / chunk_size), i.e. "
                     "however many chunks the goal is actually away. A fixed 1 would leave "
                     "far goals unreachable by construction and score them as failures.")
ap.add_argument("--extra-chunks", type=int, default=0,
                help="chunks to run BEYOND the adaptive count, to see whether the policy "
                     "settles at the goal or drifts past it")
ap.add_argument("--sector", default="",
                help="only evaluate goals in this view sector (e.g. back, front). The data is "
                     "~69%% back-facing, so 'back' is where the policy has actually been trained.")
ap.add_argument("--save-frames", action="store_true", default=True,
                help="save the observation at each chunk boundary for the report")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=30, help="flow-matching steps")
ap.add_argument("--guidance", type=float, default=1.0,
                help="1.0 = no CFG. Training used cfg_dropout_rate=0, so the model has "
                     "never seen an empty caption and a CFG uncond branch would be noise.")
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--root", default=f"{V12}/data/trajectories/v7_stage2_renders_lookat075")
ap.add_argument("--out", default=f"{V12}/runs/closed_loop/eval.json")
ap.add_argument("--resume", type=int, default=1,
                help="continue from <out>.partial.json instead of redoing "
                     "episodes a preempted run already finished")
ap.add_argument("--split", default="val", choices=("val", "train", "all"),
                help="which scenes to draw episodes from. `val` = the held-out manifest "
                     "(the generalization number). `train` = scenes the model DID train on, "
                     "which is the control the rollouts never had: without it a weak result "
                     "cannot be attributed to the room being new rather than to the policy. "
                     "`all` reproduces the old --held-out-only 0 behaviour.")
ap.add_argument("--val-scenes", default=DEFAULT_VAL_SCENES,
                help="scene manifest defining the holdout, matching the training split. "
                     "Pass '' to fall back to replaying the export enumeration, which is "
                     "how every pre-v5 number defined held-out (never-sampled placements, "
                     "whose SCENES the model had still trained on).")
ap.add_argument("--episodes-from", default="",
                help="replay a shared episode manifest (scripts/build_eval_episodes.py) instead "
                     "of drawing episodes here. This is what makes a number comparable with "
                     "another policy: the score is dominated by how far the start is "
                     "(corr(d_start, improvement) = 0.66 over 138 rollouts), so two policies "
                     "measured on two episode sets are not comparable however identical the "
                     "protocol. Empty = draw as before. NEVER give this a relative default: this "
                     "script chdirs into the vendored framework.")
ap.add_argument("--write-episodes", default="",
                help="dump the episodes this run selected as a manifest and exit — how an "
                     "existing run becomes shareable.")
ap.add_argument("--samples", type=int, default=1,
                help="draws per chunk, averaged before execution. 1 reproduces every earlier "
                     "number. K>1 targets a measured weakness: the shoot probe fired on only "
                     "62%% of windows that truly contained an arrival, and this policy's "
                     "single-draw action error is known to be dominated by sampling variance.")
ap.add_argument("--stop-on-shoot", type=int, default=1,
                help="end the rollout at the first chunk whose predicted shoot channel "
                     "crosses --shoot-threshold. 0 keeps the fixed n_chunks length, which "
                     "is what every pre-v5 number was measured with.")
ap.add_argument("--shoot-threshold", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true", help="load raw weights instead of EMA")
ap.add_argument("--policy", default="cosmos",
                choices=("cosmos", "pi05", "groot", "dp", "autophoto", "astra"),
                help="which policy answers act(frame, goal). Everything else in this script -- "
                     "renderer, episodes, stop rule, metric -- is shared, which is what makes "
                     "the numbers comparable. See src/baselines/adapters.py. For `astra`, "
                     "--checkpoint is the gateway model id (gpt-6-astra) and --chunks the query budget.")
ap.add_argument("--astra-goal-mode", default="text", choices=("text", "image"), dest="astra_goal_mode",
                help="astra: the goal as the NL shot brief or as the goal frame")
ap.add_argument("--astra-send-res", type=int, nargs=2, default=(296, 222), metavar=("W", "H"),
                dest="astra_send_res",
                help="astra: size frames and goal image are sent at. 296x222 = our 256^2 pixel "
                     "budget at the render's 4:3 aspect")
ap.add_argument("--astra-reasoning", default="low", dest="astra_reasoning",
                help="astra: reasoning_effort per query (the source blog runs every query at low)")
ap.add_argument("--limit", type=int, default=0,
                help="run only the first N episodes of whatever was selected (0 = all). For "
                     "smokes. Deliberately separate from --episodes, whose default of 24 would "
                     "otherwise silently truncate a 64-episode manifest.")
ap.add_argument("--policy-config", default="",
                help="DP: the training run's config.yaml. autophoto: the aesthetic ResNet .pt")
ap.add_argument("--goal-channel", default="prompt", choices=["prompt", "panel", "caption"],
                help="how the goal reaches a `--policy cosmos` checkpoint, and it MUST match "
                     "what that checkpoint was trained on. 'prompt' is the structured sentence "
                     "(v5, v6, the no-vision ablation); 'panel' concatenates the goal frame to "
                     "the right of the observation (camera_policy_nano_v6_goalimg); 'caption' "
                     "feeds the VLM caption of the goal frame as the text "
                     "(camera_policy_nano_v6_caption). A mismatch does not raise -- it quietly "
                     "scores a policy on a goal it was never taught to read.")
ap.add_argument("--captions", default="",
                help="with --goal-channel caption: JSON from scripts/caption_goal_frames.py "
                     "covering every goal frame in the manifest.")
ap.add_argument("--render-res", type=int, nargs=2, default=None, metavar=("W", "H"),
                help="Blender render size. Default: the smallest 4:3 multiple that squashes to "
                     "--image-size (344x258 for 256). For a cross-policy comparison use 1024 768, "
                     "the training-native render every policy here was exported from, so each "
                     "adapter's squash reproduces its own training frames exactly.")
ap.add_argument("--goal-override", default="",
                help="JSON {goals: {episode_key: {goal_vec, goal_crop}}} (scripts/build_module2_goals.py): "
                     "the goal the POLICY is told, replacing the target's ground-truth profile, while "
                     "scoring stays against the true target. How a goal read from the target photograph "
                     "(Module 2) is measured against the ground-truth row.")
ap.add_argument("--goal-image-override", default="",
                help="JSON {goals: {episode_key: {image: path}}} (scripts/build_crossref_goals.py): the "
                     "goal PICTURE the policy is shown, replacing the episode's own goal frame, while "
                     "scoring stays against the true target. How an image goal of the same shot from "
                     "another scene and subject is measured on the same episodes.")
ap.add_argument("--astra-goal-framing", default="same_view", choices=("same_view", "cross_reference"),
                dest="astra_goal_framing",
                help="astra image goals: 'same_view' tells the model to reproduce the picture (the goal "
                     "frame of this very scene); 'cross_reference' tells it the picture is from another "
                     "place and person and to compose its own subject the same way. Use the latter with "
                     "--goal-image-override: the picture can never match, so 'reproduce' is impossible.")
ap.add_argument("--render-profile", default="legacy", choices=("legacy", "v7"),
                help="'v7' renders the way the training frames were rendered (32 samples, "
                     "denoised, Nishita sky 0.1). 'legacy' keeps each scene's own world and "
                     "denoise setting, which is what every number before 2026-10-03 used and "
                     "differs from the training frame of the same pose by 24-101 /255 per "
                     "held-out scene. See src.common.run_info.RENDER_PROFILES.")
args = ap.parse_args()


def _repo_path(p: str) -> str:
    """Resolve a path argument against the REPO, never the cwd.

    This script does `os.chdir(CF_ROOT)` because Cosmos resolves model configs by relative path.
    Every path that arrives on the command line is therefore ambiguous, and the two ends of one
    can even disagree: `--out runs/closed_loop/x.json` made `write_run_info` create the run_info
    under the vendored framework while the Blender subprocess, which runs with `cwd` set to the
    shared repo, looked for it there and died `FileNotFoundError` on all five episodes.

    Fifth appearance of one bug -- it killed a training run, was fixed in the dataset module,
    came back as a `--val-scenes` default and killed three rollouts, caught the new manifest
    flags, then this. Normalising every path flag in one place is the end of it. Same rule as
    `src/data/cosmos_camera_dataset.py`.
    """
    return p if (not p or Path(p).is_absolute()) else str(Path(V12) / p)


for _flag in ("out", "root", "val_scenes", "episodes_from", "write_episodes", "checkpoint",
              "policy_config", "goal_override", "goal_image_override"):
    if _flag == "checkpoint" and args.policy == "astra":
        continue                    # a gateway model id, not a path
    setattr(args, _flag, _repo_path(getattr(args, _flag)))

# ---------------------------------------------------------------- policy
# Nothing from cosmos_framework is imported at module level any more. The cosmos adapter
# imports it (after `init_script`) when `--policy cosmos` is constructed, and the baselines'
# adapters import lerobot / transformers instead -- the two stacks cannot share a venv.
import torch  # noqa: E402

from src.baselines.adapters import make_adapter  # noqa: E402
from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.blender_env import (  # noqa: E402
    BlenderRolloutEnv, PersistentBlenderRenderer, SubprocessBlenderRenderer,
)
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.facing import sector3, sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)
from src.common.dataset_base import arrival_step  # noqa: E402
from src.common.reward import CameraIntrinsics, pose_to_geometry, _geometry_distance  # noqa: E402
from src.common.run_info import RENDER_PROFILES, renderer_kwargs  # noqa: E402
from src.common.run_info import write_run_info as _write_run_info  # noqa: E402

# Mirrors src/data/cosmos_camera_dataset.CAMERA_ACTION_DIM (see DEFAULT_VAL_SCENES above).
CAMERA_ACTION_DIM = 10

SECTOR_ORDER = ("front", "front-right", "right", "back-right",
                "back", "back-left", "left", "front-left")


def geometry_distance(position, forward, up, goal_view, intr) -> float:
    achieved = pose_to_geometry(
        position, forward, up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height,
    )
    goal = pose_to_geometry(
        goal_view.camera_position, goal_view.camera_forward, goal_view.camera_up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height,
    )
    return float(_geometry_distance(achieved, goal))


def write_run_info(placement: str, data_path: str, out_dir: Path) -> str:
    """Placement-faithful run_info; see `src.common.run_info` for why it is shared."""
    # An explicit (W, H) renders at that size; an int renders at the training aspect sized
    # for that square (src/common/run_info.render_resolution). Adapters squash from there.
    res = tuple(args.render_res) if args.render_res else args.image_size
    # RENDER ON THE GPU. `BlenderDrone` defaults to CPU with 4 threads, which write_run_info
    # calls "right for a 256x256 rollout frame and far too slow for a full-resolution one" --
    # and it is: at 1024x768 a frame took ~35 s and an episode ~2.7 min, nearly all of it
    # Cycles on the CPU while an idle B200 sat next to it. Same options the full-resolution
    # renderer already uses (scripts/render_authored_goals.py RENDER_OPTS); `samples` stays at
    # BlenderDrone's 64, which is what the training data was rendered with
    # (render_v5_from_placements.py --samples default 64), so the picture is unchanged.
    # persistent_data keeps the BVH between frames, which is the whole point of the
    # persistent server.
    opts = {"render_device": "GPU", "blender_threads": 8, "persistent_data": True,
            **RENDER_PROFILES[args.render_profile]}
    return _write_run_info(placement, data_path, out_dir, shared_root=SHARED, resolution=res,
                           extra_options=opts)


def trained_placements() -> set[str]:
    """Placement names the export actually consumed.

    Replays `scripts/export_lerobot.py`'s enumeration — same seed, same order of
    `random` consumption — and collects which placements contributed an episode.
    Verified elsewhere to reproduce the export prompt-for-prompt (4000/4000).

    This has to be computed rather than assumed. The export stops at
    `max_episodes` and takes `per_placement` episodes from each, so 4000 episodes
    come from only ~1000 of the 3931 placements; the remaining ~2931 are genuinely
    unseen. Drawing "held-out" episodes from the same shuffled directory listing —
    which this script did until now — landed on trained placements every time.
    """
    # Cache: replaying the enumeration costs ~35 min over 7.7k placements at
    # max_per_pair=24, and every shard of a split eval would otherwise pay it again.
    # Keyed by the parameters that define the trained set, so a changed export cannot
    # silently reuse a stale answer.
    r = Path(args.root)
    key = (f"{r.name}_s{args.export_seed}_m{args.export_max_episodes}"
           f"_p{args.export_per_placement}_c{args.chunk_size}")
    cache = Path(args.out).parent / f"_trained_{key}.json"
    if cache.exists():
        try:
            used = set(json.loads(cache.read_text()))
            print(f"[eval] trained set from cache: {len(used)} placements ({cache.name})",
                  flush=True)
            return used
        except Exception:  # noqa: BLE001
            pass

    placements = [(d, r / d / "data.json") for d in sorted(os.listdir(r))
                  if (r / d / "data.json").exists()]
    random.seed(args.export_seed)
    random.shuffle(placements)

    used: set[str] = set()
    n_ep = 0
    for name, path in placements:
        if n_ep >= args.export_max_episodes:
            break
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        try:
            windows = list(iter_goal_start_windows(
                path, chunk_size=args.chunk_size, max_per_pair=args.export_per_placement))
        except Exception:  # noqa: BLE001
            continue
        random.shuffle(windows)
        taken = 0
        for w in windows:
            if taken >= args.export_per_placement or n_ep >= args.export_max_episodes:
                break
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            if not all(Path(k.image).exists() for k in w.keyframes):
                continue
            used.add(name)
            n_ep += 1
            taken += 1
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(used)))
    except Exception:  # noqa: BLE001
        pass          # a cache miss is slow, not wrong
    return used


def pick_episodes() -> list:
    """Held-out (start, goal) pairs, one per placement for scene diversity.

    With `--per-sector`, coverage is balanced across the eight view sectors instead of
    following the data. That distinction matters here: 69% of well-framed goals sit
    behind the subject and only ~4% in front, so simply raising `--episodes` buys more
    back-views rather than more variety, and the per-sector numbers stay unreadable for
    exactly the sectors we most want to improve.

    Scanning stops once every sector is full OR the placement list runs out — rare
    sectors legitimately cannot fill, and the summary reports what was actually found
    rather than silently returning a lopsided set.
    """
    # comma list so one sweep can be split across GPUs by sector
    wanted = {x.strip() for x in args.sector.split(",") if x.strip()}
    dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    trained: set[str] = set()
    if args.split in ("val", "train") and args.val_scenes:
        vs = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
        want_val = args.split == "val"
        keep = [d for d in dirs if (d.split("__")[0] in vs) == want_val]
        trained = set(dirs) - set(keep)
        dirs = keep
        print(f"[eval] split={args.split}: {len(dirs)} placements "
              f"({'held out of' if want_val else 'used in'} training)", flush=True)
    elif args.held_out_only:
        if args.val_scenes:
            # A placement is held out iff its SCENE is in the val manifest -- the same
            # definition the training split uses, so "held-out" means one thing in this
            # repo instead of two. Exact and instant; the replay below is a ~35 min scan
            # that could only ever say "this placement was not sampled", which is weaker
            # (its scene, and so its whole visual environment, was still trained on).
            val_scenes = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
            keep = [d for d in dirs if d.split("__")[0] in val_scenes]
            trained = set(dirs) - set(keep)
            dirs = keep
            print(f"[eval] held-out pool: {len(dirs)} placements from "
                  f"{len(val_scenes)} val scenes ({args.val_scenes})", flush=True)
        else:
            trained = trained_placements()
            dirs = [d for d in dirs if d not in trained]
            print(f"[eval] held-out pool: {len(dirs)} placements "
                  f"({len(trained)} excluded as trained-on) [export-replay mode]", flush=True)
    # Own generator, seeded separately from the export replay above, which consumed
    # the global one.
    rng = random.Random(args.seed)
    rng.shuffle(dirs)
    picked: list = []
    per_sector: Counter = Counter()
    cap = int(args.per_sector or 0)

    for name in dirs:
        if not cap and len(picked) >= args.episodes:
            break
        if cap and len(picked) >= args.episodes:
            break
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        path = os.path.join(args.root, name, "data.json")
        if not os.path.exists(path):
            continue
        try:
            windows = list(iter_goal_start_windows(path, chunk_size=args.chunk_size,
                                                   max_per_pair=2))
        except Exception:  # noqa: BLE001
            continue
        for w in windows:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            bearing = float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
            sec = sector8(bearing)
            if wanted:
                # Exact sector8 match when the token names one, because sector3 returns
                # "front"/"side"/"back" and those two names COLLIDE with sector8's. A
                # plain membership test made `--sector back` also accept back-left and
                # back-right, which silently overlapped two supposedly disjoint jobs.
                if sec in wanted:
                    pass
                elif any(t not in SECTOR_ORDER for t in wanted) and sector3(bearing) in wanted:
                    pass
                else:
                    continue
            if cap and per_sector[sec] >= cap:
                continue          # this sector is full; try another window/placement
            picked.append((name, path, w, g, bearing, sec))
            per_sector[sec] += 1
            break

    if cap:
        got = ", ".join(f"{s}:{per_sector[s]}" for s in SECTOR_ORDER if per_sector[s])
        missing = [s for s in SECTOR_ORDER if per_sector[s] < cap]
        print(f"[eval] sector coverage (cap {cap}) -> {got}", flush=True)
        if missing:
            print(f"[eval] under cap (not enough data): "
                  f"{', '.join(f'{s}:{per_sector[s]}' for s in missing)}", flush=True)
    return picked


def episodes_from_manifest(path: str) -> list:
    """Replay a shared manifest into exactly what `pick_episodes` returns.

    A window is looked UP, not re-drawn: the manifest names `(pair_idx, start_frame_idx,
    goal_frame_idx)` and this enumerates the same pool to find it. Absent means the pool
    differs (a different root, chunk_size or max_per_pair), which is a fault to surface — a
    silent substitution would hand back a different shot under the right name.

    `d_start` is then recomputed from the window and checked against the manifest. It is pure
    pose math, so agreement proves both repos are rolling out the same episode; the check runs
    before the model is loaded, so a mismatch costs nothing.
    """
    doc = json.loads(Path(path).read_text())
    chunk, per_pair = int(doc.get("chunk_size", 8)), int(doc.get("max_per_pair", 2))
    if chunk != args.chunk_size:
        raise SystemExit(f"manifest was built with chunk_size={chunk} but --chunk-size is "
                         f"{args.chunk_size}; the window pool itself would differ")
    if doc.get("kind") in ("gs", "posed_blender"):
        return _gs_episodes(doc, path)
    picked, worst = [], 0.0
    for ep in doc["episodes"]:
        path_j = Path(ep["data_json"])
        if not path_j.exists():                     # another repo may hold the tree elsewhere
            path_j = Path(args.root) / ep["placement"] / "data.json"
        want = (ep["pair_idx"], ep["start_frame_idx"], ep["goal_frame_idx"])
        found = None
        for w in iter_goal_start_windows(path_j, chunk_size=chunk, max_per_pair=per_pair):
            if (w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) == want:
                found = w
                break
        if found is None:
            raise SystemExit(f"{ep['key']}: window {want} is not in the pool for {path_j} — "
                             "the manifest and this checkout disagree about the data")
        g = goal_vector(found.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(found))
        if not np.isfinite(g).all():
            raise SystemExit(f"{ep['key']}: goal vector is not finite here")
        d0 = geometry_distance(np.asarray(found.start.camera_position, dtype=np.float64),
                               np.asarray(found.start.camera_forward, dtype=np.float64),
                               np.asarray(found.start.camera_up, dtype=np.float64),
                               found.goal_frame, None)
        worst = max(worst, abs(d0 - float(ep["d_start"])))
        picked.append((ep["placement"], str(path_j), found, g, float(ep["bearing"]),
                       ep["sector"]))
    if worst > 1e-6:
        raise SystemExit(f"d_start disagrees with the manifest by up to {worst:.2e} — these are "
                         "not the same episodes; refusing to spend GPU on an incomparable run")
    print(f"[eval] manifest {path}: {len(picked)} episodes replayed, "
          f"d_start matches to {worst:.1e}", flush=True)
    return picked


def manifest_kind() -> str | None:
    """None for a window-lookup manifest; "gs" (3DGS placements) or "posed_blender" (Blender scenes
    with the episode's own poses and a pre-written run_info, e.g. scripts/build_spin_episodes.py)."""
    return json.loads(Path(args.episodes_from).read_text()).get("kind") if args.episodes_from else None


def _gs_episodes(doc: dict, path: str) -> list:
    """A posed manifest: each episode carries its own poses (3DGS `scripts/gs_select_episodes.py`,
    or Blender `scripts/build_spin_episodes.py`, which also carries a pre-written run_info).

    There is no data.json to look a window up in -- the builder sampled the camera path itself
    rather than reading a rendered trajectory -- so the window is rebuilt from the manifest
    into the attributes this loop reads, and the placement spec stands where the data.json
    path would: it is what the 3DGS renderer takes as its run_info. Poses are in the
    placement's local metric frame (feet at the origin, z up). The goal vector and d_start
    are recomputed from the stored label and poses and must match, as for a Blender manifest.
    """
    from types import SimpleNamespace as NS

    def pose(p):
        return tuple(np.asarray(p[k], dtype=np.float64) for k in ("position", "forward", "up"))

    picked, worst_d, worst_g = [], 0.0, 0.0
    for ep in doc["episodes"]:
        if list(ep["goal_keys"]) != list(DEFAULT_GOAL_KEYS):
            raise SystemExit(f"{ep['key']}: goal keys differ from this checkout's")
        center = np.asarray(ep["subject_center"], dtype=np.float64)
        gp, gf, gu = pose(ep["goal_pose"])
        sp, sf, su = pose(ep["start_pose"])
        goal = NS(frame_idx=ep["goal_frame_idx"], image=ep["goal_image"], raw=ep["goal_raw"],
                  camera_position=gp, camera_forward=gf, camera_up=gu,
                  subject_center=center, subject_height=float(ep["subject_height"]))
        start = NS(camera_position=sp, camera_forward=sf, camera_up=su, image=ep["start_image"],
                   object_position=[float(center[0]), float(center[1]), 0.0])
        w = NS(pair_idx=ep["traj"], start_frame_idx=ep["start_frame_idx"], goal_frame=goal,
               start=start, object=ep["object"])
        g = goal_vector(goal.raw, DEFAULT_GOAL_KEYS, object_key=ep["object"])
        if not np.isfinite(g).all():
            raise SystemExit(f"{ep['key']}: goal vector is not finite here")
        worst_g = max(worst_g, float(np.abs(g - np.asarray(ep["goal_vec"], dtype=np.float32)).max()))
        worst_d = max(worst_d, abs(geometry_distance(sp, sf, su, goal, None) - float(ep["d_start"])))
        picked.append((ep["placement"], ep.get("spec") or ep["run_info"], w, g, float(ep["bearing"]),
                       ep["sector"]))
    if worst_d > 1e-6 or worst_g > 1e-3:
        raise SystemExit(f"manifest disagrees with this checkout (d_start {worst_d:.2e}, goal "
                         f"vector {worst_g:.2e}); refusing to spend GPU on an incomparable run")
    print(f"[eval] {doc.get('kind')} manifest {path}: {len(picked)} episodes, d_start matches to "
          f"{worst_d:.1e}, goal vectors to {worst_g:.1e}", flush=True)
    return picked


def write_manifest(episodes: list, path: str) -> None:
    """Dump selected episodes in the shape `scripts/build_eval_episodes.py` writes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_eval_episodes", Path(V12) / "scripts" / "build_eval_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = [mod.materialise(name, Path(p), w, g, bearing, sec)
           for name, p, w, g, bearing, sec in episodes]
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"_what": "Episodes selected by a closed_loop_eval run, dumped for sharing.",
         "version": 1, "source_root": args.root, "val_scenes": args.val_scenes,
         "chunk_size": args.chunk_size, "max_per_pair": 2, "seed": args.seed,
         "n_episodes": len(out), "episodes": out}, indent=1))
    print(f"[eval] wrote {dest} ({len(out)} episodes)")


def timing_summary(results: list[dict]) -> dict:
    """Speed and cost over the run: policy calls, renders, episodes, and (API policies) tokens/$.

    Episodes from a partial written before timing existed have no act_s and are skipped, so the
    counts here say how many episodes the timing covers.
    """
    def stats(v):
        v = np.asarray(v, dtype=np.float64)
        if not len(v):
            return None
        return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
    timed = [r for r in results if r.get("act_s")]
    out = {"episodes_timed": len(timed),
           "act_s_per_call": stats([x for r in timed for x in r["act_s"]]),
           "render_s_per_frame": stats([x for r in timed for x in r.get("render_s", [])]),
           "calls_per_episode": stats([len(r["act_s"]) for r in timed]),
           "episode_wall_s": stats([r["episode_wall_s"] for r in timed if r.get("episode_wall_s")]),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    recs = [r["policy_record"] for r in results if r.get("policy_record")]
    if recs:
        calls = [c for p in recs for c in p["calls"]]
        api = [c["api_latency_s"] for c in calls if c["api_latency_s"]]
        out["api"] = {
            "model": recs[0]["model"], "reasoning": recs[0]["reasoning"],
            "goal_mode": recs[0]["goal_mode"], "goal_framing": recs[0]["goal_framing"],
            "send_wh": recs[0]["send_wh"],
            "n_calls": len(calls), "n_parse_failed": sum(not c["parsed"] for c in calls),
            "n_retries": sum(max(0, len(c["attempts"]) - 1) for c in calls),
            "latency_s_per_call": stats([c["latency_s"] for c in calls]),
            "api_latency_s_per_call": stats(api),
            "prompt_tokens_per_call": stats([c["prompt_tokens"] for c in calls]),
            "completion_tokens_per_call": stats([c["completion_tokens"] for c in calls]),
            "reasoning_tokens_per_call": stats([c["reasoning_tokens"] for c in calls]),
            "images_per_call": stats([c["n_images"] for c in calls]),
            "cost_usd_total": float(sum(c["cost_usd"] for c in calls)),
            "cost_usd_per_episode": stats([p["cost_usd"] for p in recs]),
            # Astra commands a velocity for dt; if a call takes longer than dt the 2 Hz control
            # loop the method assumes is not real-time -- the simulator waits for the model.
            "frac_calls_slower_than_dt": float(np.mean([c["latency_s"] > recs[0]["dt_s"] for c in calls])),
        }
    return out


def main() -> int:
    t0 = time.time()
    episodes = (episodes_from_manifest(args.episodes_from) if args.episodes_from
                else pick_episodes())
    if args.limit > 0:
        episodes = episodes[:args.limit]
        print(f"--limit {args.limit}: running the first {len(episodes)} episode(s) only", flush=True)
    if args.write_episodes:
        write_manifest(episodes, args.write_episodes)
        return 0
    print(f"[eval] {len(episodes)} episodes", flush=True)
    if not episodes:
        print("[eval] nothing to evaluate", flush=True)
        return 1

    adapter = make_adapter(args, Path(args.out).parent)
    print(f"policy: {adapter.name}  checkpoint: {args.checkpoint}", flush=True)
    print(f"[eval] policy loaded ({time.time()-t0:.0f}s)", flush=True)
    intr = CameraIntrinsics.from_render(1024, 768)
    # Resume from a previous run's partial output. This eval keeps getting preempted on
    # `share` at ~30 min, and without this every restart redoes the episodes it already
    # finished — 80 episodes at ~3.5 min each never completes in a 30-min window. With it
    # the progress accumulates across as many preemptions as it takes.
    results = []
    done_keys: set = set()
    _partial_path = Path(args.out).with_suffix(".partial.json")
    if args.resume and _partial_path.exists():
        try:
            prev = json.loads(_partial_path.read_text()).get("episodes", [])
            results = list(prev)
            # `key` when present, the old path triple otherwise, so partials written before the
            # key existed still resume.
            done_keys = {e["key"] if e.get("key") else
                         (e.get("placement"), e.get("start_frame_image"), e.get("goal_frame_image"))
                         for e in prev}
            print(f"[eval] resuming: {len(results)} episodes already done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] partial unreadable, starting fresh: {exc}", flush=True)

    # Namespaced by the output file, NOT a shared "frames" dir. Two runs writing into
    # the same directory collide on ep{i}_{tag}.jpg and silently overwrite each
    # other's renders — the numbers stay correct (they come from env poses) but the
    # saved images end up belonging to a different scene, which only shows up as a
    # subject changing mid-strip in the report.
    # SCENES whose render server dies; their placements use one process per frame. Keyed by
    # scene, not placement: Parking_70ae1f27 kills the server for every object placed in it,
    # and a placement-keyed set demoted one placement at a time while the other nine failed
    # exactly the same way -- which is how that scene has been missing from every held-out
    # number this project has published.
    _renderer_fallback: set[str] = set()
    frames_dir = Path(args.out).parent / f"frames_{Path(args.out).stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # A 3DGS manifest renders through ONE server for the whole run instead of one Blender per
    # episode: loading a splat costs ~a minute and its subject's Blender ~half that, and the
    # manifest is sorted by placement, so each is paid once per placement, not per episode.
    kind = manifest_kind()
    gs, posed = kind == "gs", kind == "posed_blender"
    gs_renderer = None
    if gs:
        from src.common.gs_render import GaussianCompositeRenderer

    overrides = json.loads(Path(args.goal_override).read_text())["goals"] if args.goal_override else None
    image_overrides = (json.loads(Path(args.goal_image_override).read_text())["goals"]
                       if args.goal_image_override else None)

    _failed = 0
    for i, (name, path, window, goal_vec, bearing, sec) in enumerate(episodes):
        _key = (f"{name}:p{window.pair_idx:02d}:s{window.start_frame_idx:02d}"
                f":g{window.goal_frame.frame_idx:02d}")
        if _key in done_keys or (name, window.start.image, window.goal_frame.image) in done_keys:
            continue
        # What the policy is TOLD. Scoring below still uses `window.goal_frame` (the true target).
        told_vec, told_raw = goal_vec, window.goal_frame.raw
        if overrides is not None:
            if _key not in overrides:
                raise SystemExit(f"--goal-override has no goal for {_key}")
            told_vec = np.asarray(overrides[_key]["goal_vec"], dtype=np.float32)
            told_raw = {**dict(window.goal_frame.raw), **overrides[_key]["goal_crop"]}
        told_image = window.goal_frame.image
        if image_overrides is not None:
            if _key not in image_overrides:
                raise SystemExit(f"--goal-image-override has no image for {_key}")
            told_image = _repo_path(image_overrides[_key]["image"])   # the eval chdirs into the framework
            if not Path(told_image).exists():
                raise SystemExit(f"--goal-image-override image missing: {told_image}")
        renderer = None
        t_ep = time.perf_counter()
        try:
            delta = abs(window.goal_frame.frame_idx - window.start_frame_idx)
            # However many chunks the goal actually is away — a fixed count would make
            # far goals unreachable by construction, then score that as policy failure.
            n_chunks = args.chunks or int(np.ceil(delta / args.chunk_size))
            n_chunks += args.extra_chunks

            # ONE Blender per episode, not one per frame. `SubprocessBlenderRenderer`
            # reloads the scene, rebuilds the BVH and recompiles kernels for EVERY render;
            # at the training render size (1024x768) that was ~95 s a frame and ~8 min an
            # episode, nearly all of it setup. The persistent server pays that once and then
            # only moves the camera (~1-2 s a frame) -- the same trade
            # `scripts/render_authored_goals.py` already makes in production.
            #
            # A scene that kills the SERVER falls back to one-process-per-frame rather than
            # losing the episode: Parking_70ae1f27 does exactly that (rc=-11), reproducibly,
            # while loading fine through the same BlenderDrone for a geometry dump.
            if gs:
                if gs_renderer is None:
                    gs_renderer = GaussianCompositeRenderer()
                run_info, renderer = path, gs_renderer
            else:
                # A posed Blender manifest brings its own run_info (e.g. the subject turned).
                run_info = path if posed else write_run_info(name, path, Path(args.out).parent / "_run_info")
                renderer = (SubprocessBlenderRenderer(
                                repo_root=SHARED, **renderer_kwargs(args.render_profile, persistent=False))
                            if name.split("__")[0] in _renderer_fallback
                            else PersistentBlenderRenderer(
                                repo_root=SHARED, **renderer_kwargs(args.render_profile, persistent=True)))
            env = BlenderRolloutEnv(run_info_path=run_info, renderer=renderer,
                                    object_position=window.start.object_position)
            start = window.start
            obs = env.reset(start.camera_position, start.camera_forward, start.camera_up)
            # Render parity for free: the start pose's training frame is on disk, so the distance
            # between it and this render says whether the policy is being shown the picture it
            # trained on (src/common/run_info.RENDER_PROFILES). None when the sizes differ.
            start_render_mae = None
            try:
                from PIL import Image
                ref = np.asarray(Image.open(start.image).convert("RGB"), dtype=np.float32)
                got = np.asarray(obs["image"], dtype=np.float32)
                if ref.shape == got.shape:
                    start_render_mae = float(np.abs(ref - got).mean())
            except Exception:  # noqa: BLE001
                pass

            d0 = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)
            trace = [d0]
            # The POSE at every scored point, not just the scalar distance. Costs nine floats
            # a chunk and makes every "what does 0.25 rad mean in degrees / metres / percent"
            # question a re-read of the results instead of another rollout -- which is exactly
            # what it cost the first time it was asked.
            def _pose_now():
                return {"position": [float(v) for v in env.position],
                        "forward": [float(v) for v in env.forward],
                        "up": [float(v) for v in env.up]}
            pose_trace = [_pose_now()]
            # Every executed STEP as well, start included. Only chunk ends are rendered and
            # scored -- 2-3 poses an episode -- which is too coarse to show the path the policy
            # flew; the trajectory GIFs re-render from this. `chunk_steps[c]` is how many steps
            # chunk c executed, so a chunk end is a prefix sum into `step_poses`.
            step_poses = [_pose_now()]
            chunk_steps: list[int] = []
            goal_text = None                # what the policy was actually told, if it says
            shots = []

            def save(tag: str, image) -> str | None:
                if not args.save_frames or image is None:
                    return None
                from PIL import Image
                out = frames_dir / f"ep{i:03d}_{tag}.jpg"
                Image.fromarray(np.asarray(image)).save(out, quality=80)
                return str(out)

            shots.append({"tag": "start", "path": save("start", obs["image"]), "d": d0})

            declared_stop = None            # chunk index where the policy said "shoot"
            stop_step = None                # step WITHIN that chunk where it crossed
            shoot_log: list[list[float]] = []
            executed_steps = 0
            # Wall time of every policy call and every render, for every policy -- the paper's
            # inference-cost table needs them and no earlier rollout recorded any. act_s is the
            # whole adapter.act (preprocessing + inference; for an API policy, the round trip).
            act_s: list[float] = []
            render_s: list[float] = []
            if hasattr(adapter, "begin_episode"):
                adapter.begin_episode(_key)
            for c in range(n_chunks):
                frame = np.asarray(obs["image"])
                # `pose` matters only to adapters whose action is defined against world axes
                # (AutoPhoto's yaw); the trained policies emit a relative chunk and ignore it.
                # goal_image: the rendered goal frame. Only the goal-channel arms read it --
                # one shows it to the policy as a second panel, the other feeds the VLM
                # caption of it -- and every other adapter ignores the kwarg.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t_act = time.perf_counter()
                chunk = adapter.act(frame, told_vec, told_raw,
                                    seed=args.seed + i * 100 + c, pose=_pose_now(),
                                    goal_image=told_image)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                act_s.append(round(time.perf_counter() - _t_act, 4))
                if goal_text is None:
                    goal_text = getattr(adapter, "last_prompt", None)
                # dim 9 is the shoot channel: a latched [0,1] state, thresholded the way
                # NVIDIA's own reference client thresholds the gripper
                # (docs/action_policy_libero_posttrain.md: "model emits [0,1]"). This is
                # the point of the 10th dim — the policy DECLARES arrival, because action
                # magnitude cannot be thresholded for it: the final chunk moves as much as
                # the previous one (0.31 vs 0.25, docs/v4_session_changes.md section 11).
                assert chunk.shape[1] == CAMERA_ACTION_DIM, (
                    f"predicted chunk is {chunk.shape[1]} wide, expected "
                    f"{CAMERA_ACTION_DIM}; the shoot channel is missing")
                # Raw channel, per step, every chunk — so a threshold sweep is a re-read
                # of the results rather than another GPU run.
                shoot_log.append([round(float(v), 5) for v in chunk[:, 9]])
                k = arrival_step(chunk[:, 9], args.shoot_threshold)
                halt = bool(args.stop_on_shoot) and declared_stop is None and k is not None
                if declared_stop is None and k is not None:
                    declared_stop, stop_step = c, k
                # Execute UP TO the declared arrival. The shoot state is latched, so only the
                # steps at and after `k` are the zero-action "you are already there" steps;
                # the ones before it are travel the policy planned, and halting on
                # `max(shoot)` discarded all of it. See `src.common.dataset_base.arrival_step`
                # — over 103 declarations the first crossing was never at step 0.
                steps = chunk[:k, :9] if halt else chunk[:, :9]
                if len(steps):
                    # BlenderRolloutEnv.step applies the 9D action via apply_action_9d, which
                    # decodes rot6d and re-projects upright — the same decode the training
                    # data was encoded with.
                    for step in steps:
                        obs, _ = env.step(step, render=False)
                        step_poses.append(_pose_now())
                    executed_steps += len(steps)
                    chunk_steps.append(len(steps))
                    _t_r = time.perf_counter()
                    obs = env.reset(env.position, env.forward, env.up)   # render the new view
                    render_s.append(round(time.perf_counter() - _t_r, 4))
                    d = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)
                    trace.append(d)
                    pose_trace.append(_pose_now())
                    shots.append({"tag": f"chunk{c+1}", "path": save(f"chunk{c+1}", obs["image"]), "d": d})
                if halt:
                    break

            d1 = trace[-1]
            d_best = float(min(trace))
            results.append({
                # Join key across repos and runs. The old identity was a triple of absolute
                # image paths, which does not survive a checkout in another location.
                "key": f"{name}:p{window.pair_idx:02d}:s{window.start_frame_idx:02d}"
                       f":g{window.goal_frame.frame_idx:02d}",
                "placement": name, "object": _window_object(window),
                "sector": sec, "bearing": bearing, "delta": delta, "chunks": n_chunks,
                # Logged even when --stop-on-shoot is off, so a fixed-length rollout still
                # answers "where WOULD it have stopped" against the same trace.
                "declared_stop": declared_stop, "executed_chunks": len(trace) - 1,
                "stop_step": stop_step, "executed_steps": executed_steps, "shoot": shoot_log,
                "d_start": d0, "d_end": d1, "d_best": d_best,
                "improvement": d0 - d1, "best_improvement": d0 - d_best,
                "trace": trace, "pose_trace": pose_trace, "shots": shots,
                "step_poses": step_poses, "chunk_steps": chunk_steps, "goal_text": goal_text,
                "start_render_mae": start_render_mae,
                # The goal this was scored against, so the units breakdown needs nothing else.
                "goal_pose": {"position": [float(v) for v in window.goal_frame.camera_position],
                              "forward": [float(v) for v in window.goal_frame.camera_forward],
                              "up": [float(v) for v in window.goal_frame.camera_up]},
                "subject_center": [float(v) for v in window.goal_frame.subject_center],
                "subject_height": float(window.goal_frame.subject_height),
                "goal_frame_image": window.goal_frame.image,
                "told_goal_image": told_image,
                "start_frame_image": start.image,
                "act_s": act_s, "render_s": render_s,
                "episode_wall_s": round(time.perf_counter() - t_ep, 3),
                # Per-call API log (latency, tokens, cost, request ids, replies) for API policies.
                "policy_record": adapter.episode_record() if hasattr(adapter, "episode_record") else None,
            })
            print(f"[eval] {i+1}/{len(episodes)} {sec:11s} delta={delta:2d} x{n_chunks} "
                  f"{name[:34]:34s} d {d0:.4f} -> {d1:.4f} (best {d_best:.4f}) "
                  f"imp {d0-d1:+.4f}", flush=True)
            # Flush after every episode. This runs on preemptible capacity and writing
            # only at the end has already cost two ~16-minute runs whose episodes were
            # complete but unrecoverable. Written via a temp file + replace so a kill
            # mid-write cannot leave a truncated JSON behind.
            _partial = Path(args.out).with_suffix(".partial.json")
            _partial.parent.mkdir(parents=True, exist_ok=True)
            _tmp = _partial.with_suffix(".tmp")
            _tmp.write_text(json.dumps(
                {"summary": {"checkpoint": args.checkpoint, "episodes": len(results),
                             "held_out_only": bool(args.held_out_only), "split": args.split,
        # A failed episode is simply absent from `results`, so a run that could not render a
        # whole scene reported a smaller n and nothing else. Measured: the August 138-episode
        # run contains ZERO Parking_70ae1f27 episodes because that scene reproducibly segfaults
        # Cycles here (rc=-11) -- so the headline v5 numbers were on seven of the eight held-out
        # scenes and no one could see it. Say what was attempted and what did not survive.
        "attempted": len(episodes), "failed": _failed,
        "scenes_completed": sorted({r["placement"].split("__")[0] for r in results}),
                             "frames_dir": str(frames_dir), "partial": True},
                 "episodes": results}, indent=1))
            _tmp.replace(_partial)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] {i+1} FAILED {name[:40]}: {type(exc).__name__}: {exc}", flush=True)
            _failed += 1
            # A 3DGS server that died or hung is replaced before the next episode; the retry
            # below then runs on the fresh one (its fallback set is never read for 3DGS).
            if gs and gs_renderer is not None and "3DGS" in str(exc):
                gs_renderer.close()
                gs_renderer = None
            # A render server that dies takes the episode with it, but the scene is usually
            # fine through one-process-per-frame (Parking_70ae1f27 is the known case). Mark
            # it so the retry, and any later window from the same placement, uses the slow
            # path instead of failing again the same way.
            if ("render server" in str(exc) or "rc=-11" in str(exc)) and \
                    name.split("__")[0] not in _renderer_fallback:
                _renderer_fallback.add(name.split("__")[0])
                print(f"[eval] {name.split('__')[0]}: render server died — that SCENE falls "
                      "back to one process per frame; re-queueing this episode", flush=True)
                # Retry this episode now on the slow path rather than losing it. Appending to
                # the list being iterated is deliberate: `enumerate` reads it lazily, so the
                # retry runs after everything already queued, by which time the scene is in
                # the fallback set. Guarded by the membership check above, so one retry only.
                episodes.append(episodes[i])
            # Abort instead of grinding through 160 identical failures. A per-episode
            # try/except is right for a bad placement, but it also swallowed a code bug that
            # broke EVERY episode: the run kept going, finished, and looked like a
            # preemption problem rather than an AttributeError on the first prediction.
            if _failed >= 5 and not results:
                print(f"[eval] ABORT — first {_failed} episodes all failed and none "
                      f"succeeded; this is a code fault, not bad data", flush=True)
                raise SystemExit(1)
        finally:
            # One Blender per EPISODE. Without this the persistent servers accumulate -- 64
            # of them, each holding a loaded scene and a GPU context.
            if renderer is not None and renderer is not gs_renderer:
                try:
                    renderer.close()
                except Exception:  # noqa: BLE001
                    pass
                renderer = None
    if gs_renderer is not None:
        gs_renderer.close()

    if not results:
        print("[eval] every episode failed", flush=True)
        return 1
    imp = np.array([r["improvement"] for r in results])
    best = np.array([r["best_improvement"] for r in results])
    d_end = np.array([r["d_end"] for r in results])
    d_best = np.array([r["d_best"] for r in results])
    d_st = np.array([r["d_start"] for r in results])
    # A radian of `reward.py` distance means nothing to a photographer, so say what it buys.
    # Calibrated in scripts/calibrate_shot_threshold.py against real frame pairs: two frames
    # this far apart carry the same six cinematography words 69% / 66% / 50% of the time.
    # `success_best` is the same at the closest point of the path, and the gap between the two
    # is the cost of not knowing when to stop.
    taus = (0.05, 0.10, 0.20)
    success = {f"success@{t:.2f}": float((d_end <= t).mean()) for t in taus}
    success_best = {f"success_best@{t:.2f}": float((d_best <= t).mean()) for t in taus}
    # Median of the per-episode ratio, not a ratio of means: improvement scales with d_start
    # (corr 0.66), and the mean is dragged by a thin negative tail (9/138 episodes, worst -2.06,
    # together worth -6.6 of a total 75.8).
    closure = 1.0 - d_end / np.maximum(d_st, 1e-9)
    by_sector = {}
    for r in results:
        by_sector.setdefault(r["sector"], []).append(r["improvement"])
    summary = {
        "by_sector": {k: {"n": len(v), "mean_improvement": float(np.mean(v))}
                      for k, v in sorted(by_sector.items())},
        "mean_best_improvement": float(best.mean()),
        "frac_best_positive": float((best > 0).mean()),
        "checkpoint": args.checkpoint,
        "episodes": len(results),
        "mean_improvement_over_noop": float(imp.mean()),
        "median_improvement": float(np.median(imp)),
        "frac_positive": float((imp > 0).mean()),
        "mean_d_start": float(np.mean([r["d_start"] for r in results])),
        "mean_d_end": float(np.mean([r["d_end"] for r in results])),
        "mean_d_best": float(d_best.mean()),
        "median_closure": float(np.median(closure)),
        **success, **success_best,
        "shot_threshold_note": "success@t = fraction whose FINAL frame is within t radians of "
                               "the goal; at t = 0.05/0.10/0.20 two real frames that close read "
                               "as the same shot 69/66/50% of the time "
                               "(runs/eval/shot_threshold.json).",
        "episodes_from": args.episodes_from,
        "held_out_only": bool(args.held_out_only), "split": args.split,
        # A failed episode is simply absent from `results`, so a run that could not render a
        # whole scene reported a smaller n and nothing else. Measured: the August 138-episode
        # run contains ZERO Parking_70ae1f27 episodes because that scene reproducibly segfaults
        # Cycles here (rc=-11) -- so the headline v5 numbers were on seven of the eight held-out
        # scenes and no one could see it. Say what was attempted and what did not survive.
        "attempted": len(episodes), "failed": _failed,
        "scenes_completed": sorted({r["placement"].split("__")[0] for r in results}),
        "frames_dir": str(frames_dir),
        "guidance": args.guidance, "num_steps": args.num_steps,
        "policy": adapter.name, "policy_config": args.policy_config or None,
        "render_res": list(args.render_res) if args.render_res else None,
        "render_profile": args.render_profile, "goal_override": args.goal_override or None,
        "goal_image_override": args.goal_image_override or None,
        "chunks": args.chunks, "ema": not args.no_ema,
        "elapsed_s": time.time() - t0,
        "timing": timing_summary(results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "episodes": results}, open(args.out, "w"), indent=1)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
