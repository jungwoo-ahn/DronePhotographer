from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.dataset import DroneActionScoreDataset
from src.vlm_qwen25.prompt import build_user_prompt
from src.vlm_qwen25.schema import parse_scores_from_text
from src.scoring.evaluator import RULE_BASED_SCORE_KEYS, CAMERA_3D_SCORE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_vl_7b_full.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="runs/qwen25_vl_eval_report.json")
    parser.add_argument("--save_predictions", action="store_true", help="save per-sample pred/gt pairs")
    return parser.parse_args()


def split_dataset_indices(length: int, eval_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if eval_ratio <= 0.0:
        return indices, []

    eval_size = int(length * eval_ratio)
    eval_size = max(1, eval_size)
    if eval_size >= length:
        eval_size = max(0, length - 1)
    if eval_size == 0:
        return indices, []
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    return train_indices, eval_indices


def torch_dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    dataset = DroneActionScoreDataset(
        annotations_path=data_cfg["annotations_path"],
        image_root=data_cfg.get("image_root"),
        action_frame=str(data_cfg.get("action_frame", "camera_local")),
        rotation_representation=str(data_cfg.get("rotation_representation", "orientation_6d")),
        distance_threshold=float(data_cfg["distance_threshold"]),
        max_pairs_per_image=int(data_cfg["max_pairs_per_image"]),
        zero_action_ratio=float(data_cfg.get("zero_action_ratio", 0.0)),
        seed=int(data_cfg["seed"]),
        target_score_keys=data_cfg.get("target_score_keys"),
    )

    _, eval_indices = split_dataset_indices(
        len(dataset),
        float(data_cfg.get("eval_ratio", 0.0)),
        int(data_cfg["seed"]),
    )
    eval_source = Subset(dataset, eval_indices) if eval_indices else dataset

    max_samples = min(int(args.max_samples), len(eval_source))
    samples = [eval_source[i] for i in range(max_samples)]

    torch_dtype = torch_dtype_from_name(model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16")))
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    score_keys = list(dataset.target_score_keys)
    bbox_keys = [k for k in score_keys if k in RULE_BASED_SCORE_KEYS]
    camera3d_keys = [k for k in score_keys if k in CAMERA_3D_SCORE_KEYS]

    abs_errors = {k: [] for k in score_keys}
    sq_errors = {k: [] for k in score_keys}
    predictions = []
    parse_failures = 0

    for sample in tqdm(samples, desc="eval"):
        user_prompt = build_user_prompt(
            str(sample["action_text"]),
            target_score_keys=score_keys,
            action_frame=dataset.action_frame,
            rotation_representation=dataset.rotation_representation,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[sample["image"]],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_text = processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
        )[0]

        pred_scores = parse_scores_from_text(generated_text, score_keys=score_keys)
        if pred_scores is None:
            parse_failures += 1
            if args.save_predictions:
                predictions.append({"pred": None, "gt": dict(sample["target_scores"]), "image": str(sample.get("image_path", ""))})
            continue

        gt_scores = sample["target_scores"]
        for key in score_keys:
            err = abs(float(pred_scores[key]) - float(gt_scores[key]))
            abs_errors[key].append(err)
            sq_errors[key].append(err * err)

        if args.save_predictions:
            predictions.append({"pred": dict(pred_scores), "gt": dict(gt_scores), "image": str(sample.get("image_path", ""))})

    # Per-key metrics
    per_key = {}
    for key in score_keys:
        mae = sum(abs_errors[key]) / len(abs_errors[key]) if abs_errors[key] else None
        rmse = (sum(sq_errors[key]) / len(sq_errors[key])) ** 0.5 if sq_errors[key] else None
        per_key[key] = {"mae": mae, "rmse": rmse}

    # Group metrics
    def _group_metrics(keys):
        all_abs = []
        all_sq = []
        for k in keys:
            all_abs.extend(abs_errors.get(k, []))
            all_sq.extend(sq_errors.get(k, []))
        mae = sum(all_abs) / len(all_abs) if all_abs else None
        rmse = (sum(all_sq) / len(all_sq)) ** 0.5 if all_sq else None
        return {"mae": mae, "rmse": rmse, "keys": keys}

    report = {
        "model_path": str(args.model_path),
        "config": str(args.config),
        "action_frame": dataset.action_frame,
        "rotation_representation": dataset.rotation_representation,
        "num_samples": len(samples),
        "parse_failures": parse_failures,
        "parse_failure_rate": 0.0 if len(samples) == 0 else parse_failures / len(samples),
        "per_key": per_key,
        "groups": {
            "bbox": _group_metrics(bbox_keys),
            "camera_3d": _group_metrics(camera3d_keys),
        },
        "total": _group_metrics(score_keys),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save predictions if requested
    if args.save_predictions and predictions:
        pred_path = output_path.with_name(output_path.stem + "_predictions.json")
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {pred_path}")

    # Print summary
    print(f"\n=== Eval Report ({len(samples)} samples, {parse_failures} parse failures) ===")
    print(f"  Total MAE: {report['total']['mae']:.4f}  RMSE: {report['total']['rmse']:.4f}")
    if bbox_keys:
        g = report["groups"]["bbox"]
        print(f"  Bbox  MAE: {g['mae']:.4f}  RMSE: {g['rmse']:.4f}  ({len(bbox_keys)} keys)")
    if camera3d_keys:
        g = report["groups"]["camera_3d"]
        print(f"  Cam3D MAE: {g['mae']:.4f}  RMSE: {g['rmse']:.4f}  ({len(camera3d_keys)} keys)")
    print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
