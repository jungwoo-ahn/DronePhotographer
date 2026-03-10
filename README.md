# DronePhotographer (Qwen2.5-VL)

This repository is focused on one workflow:

- input: `image_i` + action text (`move/rotate`)
- output: composition score JSON of `image_j`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`

## Score Families

Two scoring methods are supported:

1. `subject-aware` (VLM-scored aesthetic/composition criteria)
2. `rule-based bbox controllability` (computed from GroundingDINO detections)

Current default training uses the rule-based controllability keys:

```json
{"bbox_occupancy_ratio":0.0,"bbox_margin_top":0.0,"bbox_margin_bottom":0.0,"bbox_margin_left":0.0,"bbox_margin_right":0.0,"bbox_aspect_ratio":1.2,"bbox_centroid_offset":0.3}
```

The key order is controlled by `data.target_score_keys` in config.

## Rotation Convention

- Camera local axes for commands: `+right, +up, +forward(view direction)`.
- Stored `final_forward` and `final_up` are world-frame unit vectors.
- Default action frame is camera-local (`data.action_frame: camera_local`).
- Action rotation text is camera-local axis-angle vector (radians).
- `world` frame is still supported as a compatibility option.

## Train

```bash
python scripts/train_qwen25_vl.py --config configs/qwen25_vl_7b_full.yaml
```

2xH200 example (GPU indices `3,4`):

```bash
bash scripts/train_qwen25_vl_2_h200.sh
```

Qwen3.5-VL-9B on 2xH200:

```bash
bash scripts/train_qwen35_vl_9b_2_h200.sh
```

The launcher auto-selects a free `torchrun` master port. To force a fixed port instead:

```bash
MASTER_PORT=29601 bash scripts/train_qwen25_vl_2_h200.sh
```

If Qwen3.5-VL loading fails, upgrade `transformers` in the training environment (newer HF may be required).

`data.zero_action_ratio` controls no-action self-pairs (`image_i`, no move/no rotate action, target = `score(image_i)`).
For example, `zero_action_ratio: 0.1` targets about 10% no-action samples in the final dataset.
`data.action_frame` controls action representation (`camera_local` or `world`).

## Annotate Detections and Compute Rule-Based Scores

```bash
# 1) add GroundingDINO detections into annotations.json
python scripts/annotate_detections.py \
  --annotations_path outputs/DogWalk_v2_10k_260309_101152/annotations.json \
  --image_root outputs/DogWalk_v2_10k_260309_101152 \
  --caption "a snowman" \
  --device cuda \
  --output_path outputs/DogWalk_v2_10k_260309_101152/annotations_detected.json

# 2) compute bbox-based score_* fields from detections
python scripts/score_annotations.py \
  --annotations_path outputs/DogWalk_v2_10k_260309_101152/annotations_detected.json \
  --image_root outputs/DogWalk_v2_10k_260309_101152 \
  --output_path outputs/DogWalk_v2_10k_260309_101152/annotations_scored.json
```

## Render Smoke Test (20 Images)

```bash
bash scripts/smoke_render_object.sh
```

## Rotation Consistency Check

```bash
python scripts/check_rotation_consistency.py \
  --annotations_path "outputs/DogWalk_260215_092109/annotations copy.json" \
  --max_views 1000 \
  --pair_samples 500 \
  --strict
```

## Evaluate

```bash
python scripts/eval_qwen25_vl.py \
  --config configs/qwen25_vl_7b_full.yaml \
  --model_path runs/<run_dir>/final
```

## Single Prediction

```bash
python scripts/predict_qwen25_vl.py \
  --model_path runs/<run_dir>/final \
  --image_path outputs/DogWalk_260215_092109/images/img_0000.png \
  --action_frame camera_local \
  --action_text "move_camera_local_m(right=0.2, up=-0.1, forward=0.0); rotate_camera_local_axis_angle_rad(rx=0.0, ry=0.1, rz=-0.1)"
```
