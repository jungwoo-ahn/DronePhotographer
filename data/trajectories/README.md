# `data/trajectories/` — v7 policy training data

Drop v7 placement directories here. This is what `configs/policy/cosmos_2b.yaml`
(`data.annotation_roots`) points at, and what `CosmosDroneDataset` recursively
globs for `**/data.json`.

> The placement directories (JPEGs) are **gitignored** — only this README is
> tracked. Don't commit rendered frames; they're large (~14 MB per placement).

## Expected layout

One directory per (scene, object) placement:

```
data/trajectories/
  <scene>__<object>/                       # e.g. Abandoned-alley_9ee2b453__All-People-Are-Sisters_1795d425
    data.json                              # trajectories + per-frame bbox + V5 scores
    renders/
      pair_00_frame_00.jpg ... pair_<KK-1>_frame_31.jpg   # K_accepted × 32 JPEGs
    done.flag                              # Stage 2 (render) complete
    scored.flag                            # Stage 3 (V5 scoring) complete
  <scene>__<object>/
    ...
```

A placement is **usable for training only once `scored.flag` exists** — the goal
vector comes from `render_records[i][j].scores`. A `done.flag`-only placement has
images + bbox but no scores → the loader produces NaN goals and skips it.

## The three pipeline stages (jungwoo's v7 generator)

| Stage | Produces | Flag |
|---|---|---|
| 1 — sample | camera trajectories (`accepted_pairs[].trajectory_32f`, 32 poses each); no images | — |
| 2 — render | Blender JPEGs + per-frame mesh bbox (`render_records[][].bbox_xyxy_full`) | `done.flag` |
| 3 — score  | the 8 V5 scores per frame (`render_records[][].scores`) | `scored.flag` |

Full schema details: `src/policy/cosmos/COSMOS_API.md` § dataset, and
`docs/v7_handoff_jooyeol.md` on branch `v7_data_for_cosmos_policy`.

## Getting data onto this machine

Render happens on the GPU/Blender machines; copy finished placements here. Push
from the render machine (this box blocks outbound SSH):

```bash
# On the render machine, from .../DronePhotographer-v7/outputs
rsync -avP -e 'ssh -p 10002' \
  v7_stage2_renders/<placement_1> v7_stage2_renders/<placement_2> \
  jooyeolyun@<this-host>:/home/nas5/jooyeolyun/repos/DronePhotographer/data/trajectories/
```

Copy whole `<placement>/` directories (with the trailing-slash-free source) so
the `<placement>/data.json` + `<placement>/renders/` structure is preserved.

## If a placement is Stage-2-only (no `scored.flag`)

Run Stage 3 to populate `scores`. It needs the per-frame `bbox_xyxy_full`
(written by Stage 2) and the object rotation from the matching v6 placement JSON
(`data/vlm_object_placing_v6_*/<placement>.json`). Use jungwoo's
`scripts/v7_stage3_score.py`, or score locally with `src.scoring.compute_v5_scores`
(see the goal-space utilities in `src/policy/common/goal_space.py`).

## Resolution

The pixel-valued goal keys (`object_center_*`, `bbox_*_offset`) are normalized
against `RENDER_WIDTH`/`RENDER_HEIGHT` in `src/policy/common/goal_space.py`
(currently **1024×768**). If you render at a different resolution, update those
constants. Off-frame values are intentionally **not** clipped — a subject past
the frame edge reads as `|normalized| > 1`.
