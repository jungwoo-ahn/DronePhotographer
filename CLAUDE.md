# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DronePhotographer reframes autonomous photography as **counterfactual visuomotor policy learning**. Photography fundamentally requires imagining what a scene looks like from viewpoints the camera has not yet occupied — every framing decision is a counterfactual prediction. We train a goal-conditioned policy that jointly imagines future frames and predicts camera actions, conditioned on a structured compositional goal (shot profile).

The model is built on Cosmos (video foundation backbone) with diffusion heads for joint next-frame and next-action prediction. Training data is generated entirely in Blender from curated design-asset scenes, providing the dense counterfactual (state, action, next-state) supervision that real-world photography data cannot.

## Status: Major Pivot In Progress

The repository is mid-pivot from a **Qwen VLM forward model** (predicts next-state shot profile scores) with **MPC inference** to a **goal-conditioned Cosmos video world-action policy**. Code from the previous direction remains in-tree but is **legacy**.

See **`Cleanup.md`** for the migration plan and instructions on retiring legacy code. When in doubt about whether a file is current or legacy, check that document.

## Pipeline Stages

1. **Render** — `render_object.py` (run via Blender): curated design-asset scenes with random camera poses → RGB images, depth maps, `annotations.json`
2. **Detect** — `scripts/annotate_detections.py`: GroundingDINO bboxes → `annotations_detected.json`
3. **Profile** — `scripts/score_annotations.py`: compute shot profiles (geometric, deterministic from bboxes + camera state) → augments annotation JSON
4. **Pair** — assemble (state, action, next-state, achieved-profile) tuples via hindsight relabeling (every random transition becomes a goal-conditioned training example with the achieved profile as the goal)
5. **Train** — Cosmos policy with diffusion heads: joint next-frame + next-action prediction, conditioned on shot profile goal
6. **Infer** — direct goal-conditioned action prediction. No MPC, no planning, no inference-time optimization.

## Architecture

### Data substrate (preserved)
- Blender rendering pipeline with curated design-asset scenes
- GroundingDINO detection for subject bbox extraction
- Shot profile computation (rule-based, deterministic) — serves as the goal space
- Camera frame conventions and rotation utilities

### Policy (new direction)
- Cosmos backbone (video foundation model) with diffusion heads
- Joint prediction: next-frame appearance and next-step camera action
- Goal-conditioned on shot profile (structured, geometric)
- Hindsight relabeling at the dataset layer

## Source Layout

- `src/scenes/` — Blender scene utilities (sky shader, camera setup, render settings). **Keep.**
- `src/drones/` — `BlenderDrone` for camera/render management. **Keep.**
- `src/detectors/` — `GroundingDINODetector` wrapper. **Keep.**
- `src/scoring/` — shot profile computation. **Keep — this is now the goal space.**
  - `bbox_control.py`: 7-key profile from primary bbox geometry
  - `subject_aware.py`: 8-key composition profile (rule of thirds, lead room, etc.)
- `src/vlm_qwen25/` — **Legacy.** See `Cleanup.md`.
- `src/policy/` — **New (to be added).** Cosmos policy training, diffusion heads, hindsight-relabeling dataset, goal conditioning.
- `src/utils/` — **New (to be added).** Relocated camera/rotation utilities.

## Key Design Decisions

- **Framing**: photography as counterfactual visuomotor policy learning. The model must imagine before it acts. This is what justifies the video world-action architecture over a pure action head.
- **Goal representation**: shot profiles (geometric, deterministic, structured). Chosen because they are scene-agnostic, support natural-language steering via a separate Commentary module, and avoid the underspecification of language goals and the scene-dependence of image goals.
- **Hindsight relabeling**: every (state, random action) → achieved-profile transition becomes a (state, goal=achieved-profile, action) training tuple. Provides dense supervision without demonstrations or rewards.
- **Joint frame + action prediction**: the frame-prediction head provides scene-imagination inductive bias that pure action policies lack. Action quality is expected to improve when joint training is used.
- **No inference-time planning**: direct goal-conditioned policy. Trades inference-time flexibility (no MPC over a forward model) for deployment simplicity and connection to the VLA / world-model literature.
- **Camera conventions**: Blender local frame is +X=right, +Y=up, -Z=forward. Actions use `camera_local` by default. Stored annotation vectors (`final_forward`, `final_up`) are in world frame. Two rotation representations exist: `orientation_6d` (forward + up, 6 floats) and `rotvec` (axis-angle, 3 floats).

## Common Commands

### Environment setup
```bash
pip install -r requirements.txt
pip install --no-build-isolation -e ./repos/GroundingDINO
```

### Data pipeline (preserved, unchanged)
```bash
# Rendering (requires Blender binary at blender/blender)
bash scripts/smoke_render_object.sh          # smoke test
bash render_object.sh                         # full render

# Detection
python scripts/annotate_detections.py \
    --annotations_path outputs/<run>/annotations.json \
    --caption "dog"

# Shot profile materialization
python scripts/score_annotations.py \
    --annotations_path outputs/<run>/annotations_detected.json
```

### Policy training (new — to be implemented)
```bash
# Cosmos policy training (placeholder — actual entry point TBD)
python scripts/train_policy.py --config configs/cosmos_policy.yaml
```

## Output Directories

- `outputs/<run_name>/` — rendered images, depth maps, annotation JSONs
- `runs/<timestamp>_<run_name>/` — training outputs (config, checkpoints, final model, summary)
- `repos/` — external repos (e.g., GroundingDINO); gitignored
- `blender/` — Blender binary; gitignored

## Notes for Working in This Repo

- The shot profile is now the **goal space**, not the model output. Treat any code that frames shot profiles as "predictions" or "scores to be regressed against" as legacy.
- The model no longer predicts JSON. New training data is (image, goal-profile, action) triples for goal-conditioned policy learning, with the model producing an action distribution and a next-frame prediction.
- Inference is a single forward pass per action decision. There is no longer a planning loop, no MPC, no candidate-action sampling at inference time. If you find yourself wanting to add one, that's a signal to check whether you're reverting to the old paradigm.
- The conceptual contribution of the project is the **framing** (photography as counterfactual visuomotor policy) plus the **dataset** (Blender curated scenes with hindsight-relabeled transitions). The architecture is downstream. Don't lose sight of this when writing code, configs, or experiments.
