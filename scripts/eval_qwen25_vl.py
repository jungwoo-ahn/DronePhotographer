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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_vl_7b_full.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_path", type=str, default="runs/qwen25_vl_eval_report.json")
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

    torch_dtype = torch_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    score_keys = list(dataset.target_score_keys)
    abs_errors = {k: [] for k in score_keys}
    sq_errors = {k: [] for k in score_keys}
    parse_failures = 0

    for sample in tqdm(samples, desc="eval"):
        user_prompt = build_user_prompt(
            str(sample["action_text"]),
            target_score_keys=score_keys,
            action_frame=dataset.action_frame,
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
            continue

        gt_scores = sample["target_scores"]
        for key in score_keys:
            err = abs(float(pred_scores[key]) - float(gt_scores[key]))
            abs_errors[key].append(err)
            sq_errors[key].append(err * err)

    report = {
        "model_path": str(args.model_path),
        "action_frame": dataset.action_frame,
        "num_samples": len(samples),
        "parse_failures": parse_failures,
        "parse_failure_rate": 0.0 if len(samples) == 0 else parse_failures / len(samples),
        "score_keys": score_keys,
        "metrics": {},
    }

    for key in score_keys:
        mae = sum(abs_errors[key]) / len(abs_errors[key]) if abs_errors[key] else None
        rmse = (sum(sq_errors[key]) / len(sq_errors[key])) ** 0.5 if sq_errors[key] else None
        report["metrics"][key] = {"mae": mae, "rmse": rmse}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
