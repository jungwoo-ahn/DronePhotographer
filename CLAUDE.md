# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DronePhotographer generates synthetic drone-camera views in Blender, annotates them with GroundingDINO object detections, and trains a Qwen vision-language model to predict composition scores given an image and a camera action. The model outputs are JSON score predictions, not classification or regression — training masks prompt tokens and only optimizes on the JSON response.

## Pipeline Stages

1. **Render** — `render_object.py` (run via Blender): random camera poses → RGB images + `annotations.json`
2. **Detect** — `scripts/annotate_detections.py`: add GroundingDINO bboxes → `annotations_detected.json`
3. **Score** (optional) — `scripts/score_annotations.py`: materialize rule-based scores into annotation JSON
4. **Train** — `scripts/train.py --config configs/<config>.yaml`: fine-tune Qwen VLM with DeepSpeed
5. **Eval/Infer** — `scripts/eval_qwen25_vl.py`, `scripts/predict_qwen25_vl.py`, `scripts/infer_mpc_blender.py`

## Common Commands

```bash
# Environment setup
pip install -r requirements.txt
pip install --no-build-isolation -e ./repos/GroundingDINO

# Rendering (requires Blender binary at blender/blender)
bash scripts/smoke_render_object.sh          # smoke test
bash render_object.sh                         # full 10k render

# Training
python scripts/train.py --config configs/qwen25_vl_7b_full.yaml           # single GPU
MASTER_PORT=29601 bash scripts/train_qwen25_vl_2_h200.sh                  # multi-GPU

# Evaluation
python scripts/eval_qwen25_vl.py --config configs/qwen25_vl_7b_full.yaml --model_path runs/<run>/final

# Single prediction
python scripts/predict_qwen25_vl.py --model_path runs/<run>/final --image_path <img> --action_text "..."

# Detection annotation
python scripts/annotate_detections.py --annotations_path outputs/<run>/annotations.json --caption "dog"
```

## Architecture

### Source layout (`src/`)

- **`vlm_qwen25/`** — VLM training core
  - `dataset.py`: `DroneActionScoreDataset` — builds image-action-score pairs on-the-fly using camera distance thresholds; splits train/eval via `eval_ratio`
  - `collator.py`: `QwenVLScoreCollator` — formats chat messages (image + instruction → JSON), masks prompt tokens in labels
  - `prompt.py`: builds action text strings and user prompts; routes between `rotvec` and `orientation_6d` representations
  - `schema.py`: canonical score JSON serialization/parsing with fixed key ordering
  - `rotation_utils.py`: camera frame conversions (world ↔ camera-local), Gram-Schmidt orthonormalization, Blender camera conventions
  - `mpc.py`, `objective.py`: model predictive control for inference-time composition optimization
- **`detectors/`** — `GroundingDINODetector` wrapper with `Detection` dataclass
- **`scoring/`** — two score families:
  - Rule-based (`bbox_control.py`): 7 keys derived from primary bbox geometry (occupancy, margins, aspect ratio, centroid offset)
  - Subject-aware (`subject_aware.py`): 8 composition keys (rule of thirds, lead room, etc.) — require pre-computed `score_<key>` fields
- **`drones/`** — `BlenderDrone` class for Blender scene/camera/render management
- **`scenes/`** — Blender scene utilities (sky shader, camera setup, render settings)

### Key design decisions

- **Pair construction is lazy**: `DroneActionScoreDataset` builds source-target pairs at init time from camera positions within `distance_threshold`. Rule-based scores can be computed on the fly from detections without pre-materialization.
- **Loss masking**: The collator sets `labels=-100` for all prompt tokens so the model only learns to generate the JSON score output.
- **Camera conventions**: Blender uses local +X=right, +Y=up, -Z=forward. Actions use `camera_local` frame by default. Stored annotation vectors (`final_forward`, `final_up`) are in world frame.
- **Two rotation representations**: `orientation_6d` (forward + up vectors, 6 floats) and `rotvec` (axis-angle, 3 floats). Config selects which via `data.rotation_representation`.

## Config Structure (YAML)

Configs live in `configs/`. Key sections:
- `model`: HuggingFace model path, dtype, attention impl
- `data`: annotations path, image root, action frame, rotation repr, score keys, pairing params (`distance_threshold`, `max_pairs_per_image`, `zero_action_ratio`, `eval_ratio`)
- `training`: output root, run name, epochs, batch size, gradient accumulation, LR, warmup, max sequence length, DeepSpeed config path

## Output Directories

- `outputs/<run_name>/` — rendered images, depth maps, annotation JSONs
- `runs/<timestamp>_<run_name>/` — training outputs (config, checkpoints, final model + processor, summary)
- `repos/` — external repos (e.g., GroundingDINO); gitignored
- `blender/` — Blender binary; gitignored
