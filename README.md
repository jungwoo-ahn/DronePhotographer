# DronePhotographer (Qwen2.5-VL)

This repository is focused on one workflow:

- input: `image_i` + action text (`move/rotate`)
- output: composition score JSON of `image_j`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`

## Score Keys

The model predicts this fixed JSON schema:

```json
{"rule_of_thirds_line":0.0,"breathing_space":0.0,"centeredness":0.0,"subject_size_20":0.0,"subject_size_80":0.0}
```

## Train

```bash
python scripts/train_qwen25_vl.py --config configs/qwen25_vl_7b_full.yaml
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
  --action_text "move(x=0.2, y=-0.1, z=0.0); rotate_axis_angle(rx=0.0, ry=0.1, rz=-0.1)"
```
