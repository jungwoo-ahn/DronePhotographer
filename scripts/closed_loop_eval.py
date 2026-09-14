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
from src.data.cosmos_camera_dataset import DEFAULT_VAL_SCENES  # noqa: E402

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


for _flag in ("out", "root", "val_scenes", "episodes_from", "write_episodes", "checkpoint"):
    setattr(args, _flag, _repo_path(getattr(args, _flag)))

# ---------------------------------------------------------------- policy
from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()  # must precede other cosmos imports

import torch  # noqa: E402

from cosmos_framework.data.generator.action.json_formatter import (  # noqa: E402
    ActionPromptJsonFormatter,
)
from cosmos_framework.inference.action import build_action_batch  # noqa: E402
from cosmos_framework.inference.args import ModelMode, OmniSetupOverrides  # noqa: E402
from cosmos_framework.inference.common.init import init_output_dir  # noqa: E402
from cosmos_framework.inference.inference import OmniInference  # noqa: E402
from cosmos_framework.scripts.action_policy_server_utils import (  # noqa: E402
    maybe_init_distributed,
)

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.facing import sector3, sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)
from src.common.dataset_base import arrival_step  # noqa: E402
from src.common.reward import CameraIntrinsics, pose_to_geometry, _geometry_distance  # noqa: E402
from src.common.run_info import to_policy_frame  # noqa: E402
from src.common.run_info import write_run_info as _write_run_info  # noqa: E402
from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402

SECTOR_ORDER = ("front", "front-right", "right", "back-right",
                "back", "back-left", "left", "front-left")


def load_policy():
    maybe_init_distributed()
    wan = os.environ.get(
        "WAN_VAE_PATH",
        f"{V12}/repos/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
    )
    overrides = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint,
        "config_file": "cosmos_framework/configs/base/config.py",
        "experiment": "action_policy_camera_nano",
        "experiment_overrides": [f"model.config.tokenizer.vae_path={wan}"],
        "output_dir": str(Path(args.out).parent / "_infer"),
        "sampler": "unipc",
        "guardrails": False,          # a rollout loop must not pay for the safety models
        "use_ema_weights": not args.no_ema,
    })
    setup = overrides.build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec: np.ndarray, crop=None, *, idle_frames: int | None = 0) -> str:
    """The goal as the JSON string the model was trained on.

    `crop` is not optional in practice: the exporter always passes it
    (`export_lerobot.py` -> `goal_prompt(g, crop=w.goal_frame.raw)`), so every training
    prompt ends with a crop phrase and a `visible` number. Omitting it here made every
    rollout so far ask the policy with a prompt shape it had never seen.

    Training prompts carry `idle_frame`, and the stock inference path drops it, so
    pass mode/idle_frames explicitly — otherwise the prompt drifts from training in
    a way that is invisible until the numbers come out wrong.

    `idle_frames` is a knob because that field LEAKS the shoot label. Measured over 6000
    exported episodes: idle_frames > 0 => shoot=1 in 896/896 (100%), idle_frames == 0 =>
    11.3%. It cannot be otherwise — post-arrival steps have translation exactly 0 and rot6d
    exactly identity, so they are idle by construction. 0 is the training-modal value (81%);
    `None` omits the field entirely, which is the ablation that separates "the policy learned
    to declare arrival" from "the policy is reading the leak".
    """
    formatter = ActionPromptJsonFormatter()
    # __call__ takes the sample dict and REPLACES ai_caption with the JSON structure,
    # reading fps/size/idle from the same keys the dataset uses — hence `viewpoint`
    # and `conditioning_fps`, not `view_point`/`fps`.
    sample = {
        "ai_caption": goal_prompt(goal_vec, crop=crop),
        "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
        "image_size": torch.tensor([args.image_size, args.image_size]),
        "mode": "policy",
        # `action` is here ONLY so the formatter can count total frames: `_get_total_frames`
        # reads action.shape[0], and without it idle_frame reads "0." where training says
        # "0 out of 8." — a 9-character divergence that string-level inspection missed and
        # only a field-by-field diff against the dataset's own output surfaced. Zeros are
        # correct: policy mode never conditions on the action, only on its length.
        "action": torch.zeros(args.chunk_size, 1, dtype=torch.float32),
        "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                             dtype=torch.uint8),
    }
    if idle_frames is not None:
        sample["idle_frames"] = torch.tensor(int(idle_frames))
    out = formatter(sample)["ai_caption"]
    return out if isinstance(out, str) else json.dumps(out)


def predict_chunk(model, frame_uint8: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    """One (chunk_size, CAMERA_ACTION_DIM) action chunk: 9 pose dims + shoot on dim 9.

    The batch is rebuilt every call on purpose: the model normalizes video in place
    and stamps `is_preprocessed`, so a reused dict silently takes the float branch.
    """
    # The render is at the training ASPECT (4:3); squash it to the square the policy
    # was trained on. A native square render would widen the vertical FOV by 4/3.
    frame_uint8 = to_policy_frame(frame_uint8, args.image_size)
    video = torch.from_numpy(frame_uint8).permute(2, 0, 1).contiguous()      # [C,H,W]
    video = video.unsqueeze(1).repeat(1, args.chunk_size + 1, 1, 1)          # [C,T,H,W]
    batch = build_action_batch(
        video=video,
        action=torch.zeros(args.chunk_size, 64, dtype=torch.float32),
        raw_action_dim=CAMERA_ACTION_DIM,
        prompt=prompt,
        view_point="ego_view",
        domain_name=DOMAIN_NAME,
        # ModelMode, not the bare string: `build_action_batch` reads `model_mode.value`.
        # ModelMode is a StrEnum, so "policy" compares equal and every type check passes
        # — it only blows up inside the builder, once per episode.
        model_mode=ModelMode.POLICY,
        action_chunk_size=args.chunk_size,
        fps=args.fps,
        input_video_key=model.config.input_video_key,
        batch_size=1,
        device="cuda",
    )
    # `build_action_batch` runs `_format_prompt` on whatever it is handed, and `prompt` is
    # ALREADY the formatted JSON from `make_prompt`. That formatted it a second time: the
    # model saw an 868-char JSON whose actions[0].description was the entire escaped
    # 573-char first JSON, a shape training never produced. Training formats exactly once
    # (the dataset transform), so overwrite the field rather than let it be wrapped again.
    batch["ai_caption"] = [prompt] * len(batch["ai_caption"])

    with torch.no_grad():
        draws = []
        for k in range(max(1, args.samples)):
            r = model.generate_samples_from_batch(
                batch, guidance=args.guidance, num_steps=args.num_steps,
                seed=[seed + 100_000 * k],
            )
            draws.append(r["action"][0].float().cpu().numpy())
    # Mean of K. Averaging the SHOOT dim too is deliberate: its output is effectively binary
    # (measured on 300 held-out windows: 90.5% below 0.05, 9.5% above 0.95, 0.1% between), so
    # the mean is the FRACTION OF DRAWS that voted to stop and thresholding it at 0.5 is a
    # majority vote, not a blurred number.
    # `draws` are already numpy (`.float().cpu().numpy()` above), so the old
    # `out["action"][0].float()...` tail cannot apply to them -- that mismatch made EVERY
    # episode raise AttributeError and be recorded as FAILED, while the run itself kept
    # going and looked like a preemption problem.
    chunk = (draws[0] if len(draws) == 1 else np.mean(draws, axis=0))
    # Slice to the FULL action width, not 9. Truncating here dropped the shoot channel
    # before the caller ever saw it, and the caller's `if chunk.shape[1] > 9` guard then
    # quietly read 0.0 forever -- termination would simply never fire, with nothing in the
    # log to say why. Same silent-fallback shape as the diagnostics collector bug.
    chunk = chunk[:, :CAMERA_ACTION_DIM]
    assert chunk.shape[1] == CAMERA_ACTION_DIM, (
        f"predicted chunk is {chunk.shape[1]} wide, expected {CAMERA_ACTION_DIM}")
    return chunk


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
    return _write_run_info(placement, data_path, out_dir,
                           shared_root=SHARED, resolution=args.image_size)


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


def main() -> int:
    t0 = time.time()
    episodes = (episodes_from_manifest(args.episodes_from) if args.episodes_from
                else pick_episodes())
    if args.write_episodes:
        write_manifest(episodes, args.write_episodes)
        return 0
    print(f"[eval] {len(episodes)} episodes", flush=True)
    if not episodes:
        print("[eval] nothing to evaluate", flush=True)
        return 1

    model = load_policy()
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
    frames_dir = Path(args.out).parent / f"frames_{Path(args.out).stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _failed = 0
    for i, (name, path, window, goal_vec, bearing, sec) in enumerate(episodes):
        _key = (f"{name}:p{window.pair_idx:02d}:s{window.start_frame_idx:02d}"
                f":g{window.goal_frame.frame_idx:02d}")
        if _key in done_keys or (name, window.start.image, window.goal_frame.image) in done_keys:
            continue
        try:
            delta = abs(window.goal_frame.frame_idx - window.start_frame_idx)
            # However many chunks the goal actually is away — a fixed count would make
            # far goals unreachable by construction, then score that as policy failure.
            n_chunks = args.chunks or int(np.ceil(delta / args.chunk_size))
            n_chunks += args.extra_chunks

            renderer = SubprocessBlenderRenderer(repo_root=SHARED)
            run_info = write_run_info(name, path, Path(args.out).parent / "_run_info")
            env = BlenderRolloutEnv(run_info_path=run_info, renderer=renderer,
                                    object_position=window.start.object_position)
            start = window.start
            obs = env.reset(start.camera_position, start.camera_forward, start.camera_up)

            d0 = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)
            trace = [d0]
            shots = []

            def save(tag: str, image) -> str | None:
                if not args.save_frames or image is None:
                    return None
                from PIL import Image
                out = frames_dir / f"ep{i:03d}_{tag}.jpg"
                Image.fromarray(np.asarray(image)).save(out, quality=80)
                return str(out)

            shots.append({"tag": "start", "path": save("start", obs["image"]), "d": d0})

            prompt = make_prompt(goal_vec, crop=window.goal_frame.raw)
            declared_stop = None            # chunk index where the policy said "shoot"
            stop_step = None                # step WITHIN that chunk where it crossed
            shoot_log: list[list[float]] = []
            executed_steps = 0
            for c in range(n_chunks):
                frame = np.asarray(obs["image"])
                chunk = predict_chunk(model, frame, prompt, seed=args.seed + i * 100 + c)
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
                    executed_steps += len(steps)
                    obs = env.reset(env.position, env.forward, env.up)   # render the new view
                    d = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)
                    trace.append(d)
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
                "trace": trace, "shots": shots,
                "goal_frame_image": window.goal_frame.image,
                "start_frame_image": start.image,
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
            # Abort instead of grinding through 160 identical failures. A per-episode
            # try/except is right for a bad placement, but it also swallowed a code bug that
            # broke EVERY episode: the run kept going, finished, and looked like a
            # preemption problem rather than an AttributeError on the first prediction.
            if _failed >= 5 and not results:
                print(f"[eval] ABORT — first {_failed} episodes all failed and none "
                      f"succeeded; this is a code fault, not bad data", flush=True)
                raise SystemExit(1)

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
        "chunks": args.chunks, "ema": not args.no_ema,
        "elapsed_s": time.time() - t0,
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
