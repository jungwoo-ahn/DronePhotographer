# Comparing policies: what has to be shared, and what already is

## SUPERSEDED, 2026-09-14 — the baselines now run HERE

This was written to hand the manifest to jooyeol so the baselines could be run on their side
and the `summary.json` files sent back. That is no longer the plan, for two reasons that
arrived together:

* **The checkpoints are local.** `baseline_ckpts/` holds pi0.5 @300k, GR00T @210k and the
  Diffusion Policy `ckpt_best.pt`, copied with sha256 verification (`PROVENANCE.md`).
* **Two bugs invalidated every closed-loop number on both sides** (commit `e766c09`): the stop
  rule discarded a mean 5.8 of 8 steps on any chunk that declared arrival, and eval rendered
  256x256 natively where training squashed 1024x768, making the subject 0.75x too small on
  every frame. Their harness is a port of ours and carries both. Re-running under a corrected
  port would mean trusting that the port stayed faithful through the fix as well.

So instead of two harnesses agreeing, there is now **one harness with four adapters**:
`scripts/closed_loop_eval.py --policy {cosmos,pi05,groot,dp}`. The renderer, the episodes, the
stop rule, the horizon and the metric code are literally the same objects; the only
policy-specific thing left is `act(frame, goal_vec, goal_raw, seed) -> (8, 10)`, in
`src/baselines/adapters.py`. The comparison is then true by construction rather than by audit.

Submit them all with `scripts/submit_policy_comparison.sh`; join with `compare_policies.py`.

**Everything below still holds** — it is why the manifest exists and what the distance means.
The parts about editing `closed_loop_eval_baselines.py` and mailing results back do not apply.

## The short version

The protocol is already aligned; the **episodes** are not, and that is the whole difference.
`configs/eval_episodes_v1.json` fixes it: 64 episodes, materialised, with a checksum that makes a
mismatch impossible to miss.

## What is already the same (checked, not assumed)

`scripts/closed_loop_eval_baselines.py` on `origin/main` (PR #30) is a faithful port:

| | |
|---|---|
| distance | `pose_to_geometry` + `_geometry_distance`, radians — your `reward.py` is byte-identical to ours, so nothing was ported |
| headline | `improvement = d_start − d_end`; a no-op scores exactly 0 |
| horizon | `ceil(delta / chunk_size) + extra_chunks`, chunk 8, extra 0 |
| shoot-stop | threshold 0.5, stop **before** executing the chunk |
| rendering | at chunk boundaries only, never between steps |
| action | 10-D, `apply_action_9d`, dim 9 = latched shoot |
| goal key | `subject_bearing_deg` (subject-relative, not the world azimuth) |
| val split | your `configs/policy/val_scenes.json` is byte-identical to our `configs/val_scenes.json` — the same eight scenes |
| gates | `GOAL_OCCUPANCY_RANGE (10,80)`, `DELTA_RANGE (0,32)`, `NEAR_FRACTION 0.25`, `MIN_GOAL_VISIBLE_FRAC 0.35` — all four match |

Nothing in that list needs to change.

## What is different, and why it decides the numbers

**1. The score is dominated by how far the start is.** Over 138 held-out rollouts,
`corr(d_start, improvement) = 0.66`, slope 0.82. Restricting our episodes to your d_start band
moved our own mean improvement from 0.549 to 0.270 — same policy, same checkpoint, same protocol.
Two policies measured on two episode sets are therefore not comparable, however identical
everything else is.

**2. The first-N-alphabetically rule lands in one room.** `select_placements` takes the first N of
`sorted(os.listdir(root))` that are in the val manifest. `Chill-Camping_150d7263` sorts first among
the eight scenes and has 60 placements, so the first 8 episodes are all from that one scene —
reproduced on our tree, 8/8. "Held out" is a claim about unseen *rooms*; that set exercises one of
the eight.

**3. Different window pool.** Our `pick_episodes` enumerates
`iter_goal_start_windows(..., max_per_pair=2)`; yours uses the uncapped pool and takes
`next(iter(...))`. Even the same placement list would not give the same episodes, which is why the
manifest pins `chunk_size` and `max_per_pair` rather than sharing placement names.

**4. n = 8 cannot rank anything.** Bootstrapping 8-episode draws from our 138 gives a 5–95 % range
of **[0.21, 0.84]** for a single policy. Your three sit at 0.284 / 0.409 / 0.455 — inside that band
end to end. The manifest is 64 episodes for this reason.

**5. Still open: what is `data/trajectories_full`?** Applying your selection rule to our
`v7_stage2_renders_lookat075` gives `mean d_start = 0.516`, and to the legacy `data/trajectories`
0.016 — neither is your 0.721. Some input still differs. You do not have to answer this before
running: the manifest's checksum turns it from a silent confound into a loud failure.

## The manifest

`configs/eval_episodes_v1.json` — 64 episodes, 8 per view sector, spread over all eight held-out
scenes (`Modern-Warehouse 12 / basement 11 / Parking 10 / Chill-Camping 9 /
office-building 9 / Nature-Snowy 6 / Modern-Office 5 / Desert-Roadside 2`), one per placement,
`mean d_start = 0.866`, delta 8–31, 1–4 chunks.

Every episode is **materialised**, so you never have to reproduce our window enumeration:

```
key, placement, scene, object, data_json, pair_idx, start_frame_idx, goal_frame_idx,
start_pose{position,forward,up}, goal_pose{...}, subject_center, subject_height,
goal_raw{...}, goal_vec, goal_keys, start_image, goal_image,
delta, n_chunks, sector, bearing, d_start
```

`goal_raw` is the goal frame's scores as the loader enriches them (visible-bbox geometry + crop
fractions) — the *inputs* to conditioning, not a prompt. You build your prompt from it with your
own `goal_text.goal_prompt`, DP normalises its own vector, we build ours with
`src/goal_authoring/serialize.goal_prompt`. The manifest does not take sides.

**`d_start` is the checksum.** It is pure pose math from `start_pose` / `goal_pose` /
`subject_center` / `subject_height`:

```python
a = pose_to_geometry(**start_pose, subject_center=sc, subject_height=sh)
g = pose_to_geometry(**goal_pose,  subject_center=sc, subject_height=sh)
assert abs(_geometry_distance(a, g) - ep["d_start"]) < 1e-6
```

Run that over all 64 before loading a model. If it passes, the two repos are provably rolling out
the same shots and question 5 above stops mattering. If it fails, stop — the run would not be
comparable, and finding that out afterwards costs a day.

## Getting the file

`configs/eval_episodes_v1.json`, 177 KB. Two ways, whichever is less trouble:

- it is on the B200 NAS at
  `/home/nas_main/jungwooahn/projects/DronePhotographerV12/configs/eval_episodes_v1.json`,
  which your cluster account can read directly;
- or pull it from `jungwoo-ahn/DronePhotographerV12`, branch `002-goal-authoring`.

It has to end up wherever you run the baselines, so if that is the yj box, copy it across.

**The absolute paths in it are ours.** `data_json`, `start_image` and `goal_image` point into
`/home/nas_main/jungwooahn/projects/DronePhotographerV12/data/trajectories/v7_stage2_renders_lookat075/`.
Rebuild them from your own root — every episode carries `placement`, so
`<your_root>/<placement>/data.json` is all you need, and the v6 placement JSON keeps the name it
already has. Nothing else in the manifest is path-dependent.

## What to change on your side

In `scripts/closed_loop_eval_baselines.py`, replace the `select_placements` + `next(iter(...))`
step with a manifest reader. Two ways, both fine:

- **Poses only** (no dependency on your window enumeration): take `start_pose` as the rollout's
  initial pose and build the goal view from `goal_pose` + `subject_center` + `subject_height`;
  `geometry_distance` needs nothing else. `n_chunks` comes from the manifest, not recomputed.
- **Look the window up**: enumerate
  `iter_goal_start_windows(dj, chunk_size=8, max_per_pair=2)` and match
  `(pair_idx, start_frame_idx, goal_frame.frame_idx)`. This is what our
  `closed_loop_eval.episodes_from_manifest` does; it also requires your tree to be the same one.

Either way, echo the manifest's `key` into each result dict — that is what the comparison joins
on. One warning from our side: this script `chdir`s into a vendored checkout, so a *relative*
manifest path resolves in the wrong place. It caught us on the first run; resolve against the repo
root (`closed_loop_eval._repo_path`).

Nothing about your reporting has to change: `d_start` / `d_end` / `d_best` / `improvement` per
episode is everything the comparison needs.

## What to run

Three runs, your existing flags plus the manifest. The manifest fixes the episode set and the
horizon, so `--episodes` and the placement selection no longer apply:

```bash
PYTHONPATH=. python scripts/closed_loop_eval_baselines.py --policy pi05 \
    --checkpoint runs/pi05_fair/checkpoints/190000/pretrained_model \
    --episodes-from eval_episodes_v1.json --out runs/shared/pi05

PYTHONPATH=. python scripts/closed_loop_eval_baselines.py --policy dp \
    --checkpoint <ckpt_best> --config configs/policy/diffusion_policy_dinov2.yaml \
    --episodes-from eval_episodes_v1.json --out runs/shared/dp

PYTHONPATH=. python scripts/closed_loop_eval_baselines.py --policy groot \
    --checkpoint runs/groot/checkpoints/130000/pretrained_model \
    --episodes-from eval_episodes_v1.json --out runs/shared/groot
```

Keep `--stop-on-shoot 1`, `--shoot-threshold 0.5`, `--extra-chunks 0`, `--chunk-size 8` at their
defaults — those already match what we run. 64 episodes at roughly a Blender scene load plus a few
chunks each is on the order of an hour or two per policy.

We are running `camera_policy_nano_v5 @ iter_000040000` on the same manifest, twice: once at
`--stop-on-shoot 1 --samples 1`, which is the row that lines up with yours, and once at
`--samples 4` (averaging four draws per chunk), reported separately as extra inference compute
rather than mixed in.

## What comes back

Three `summary.json` files. Then:

```
python scripts/compare_policies.py \
    --run ours=runs/closed_loop/v5_shared_stopshoot.json \
    --run pi05=<yours> --run dp=<yours> --run groot=<yours> --ref ours
```

It inner-joins on `key`, **refuses** if any shared key's `d_start` disagrees by more than 1e-6, and
prints per-policy rows plus paired per-episode differences with a bootstrap CI and a sign test —
which a shared episode set is what makes possible, and which is far more sensitive than comparing
two means.

## One more thing: what the distance means

`improvement` says how much closer the camera got. It does not say whether the requested photograph
was taken. `scripts/calibrate_shot_threshold.py` measures that directly: over 4,800 pairs of real
frames within a placement (held-out scenes), how often do two frames that far apart carry the same
six cinematography words — shot size, view sector, elevation band, placement x/y, crop side?

| distance | pairs reading as the same shot |
|---|---|
| < 0.05 | 69 % |
| < 0.10 | 66 % |
| < 0.20 | 50 % |
| > 0.60 | 0 % |

Our v5 @40K ends at `d_end` ≈ 0.35–0.40, so **17–25 % of rollouts finish inside 0.10** and 6–9 %
inside 0.05, against a median closure of 72–74 % that reads like success. Your π0.5 at
`d_end = 0.266` is in the same regime. Worth reporting `success@tau` alongside `improvement` for
all four policies — the comparison script computes it from fields you already emit.

And a hint about where the headroom is: our `d_best` (closest point on the path) reaches 0.10 on
38 % of episodes against 17 % at the end. The policy passes near the shot and keeps going. A
perfect stopping rule is worth roughly a doubling.
