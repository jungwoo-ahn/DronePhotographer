from __future__ import annotations

import argparse
import math
import inspect
import json
import os
import random
from datetime import datetime
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import Subset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.collator import QwenVLScoreCollator
from src.vlm_qwen25.dataset import DroneActionScoreDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_vl_7b_full.yaml")
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


def compute_warmup_steps(
    *,
    train_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
    total_training_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    return max(0, round(total_training_steps * warmup_ratio))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["data"]["seed"])
    set_seed(seed)
    action_frame = str(cfg["data"].get("action_frame", "camera_local"))
    rotation_representation = str(cfg["data"].get("rotation_representation", "orientation_6d"))

    dataset = DroneActionScoreDataset(
        annotations_path=cfg["data"]["annotations_path"],
        image_root=cfg["data"].get("image_root"),
        action_frame=action_frame,
        rotation_representation=rotation_representation,
        distance_threshold=float(cfg["data"]["distance_threshold"]),
        max_pairs_per_image=int(cfg["data"]["max_pairs_per_image"]),
        zero_action_ratio=float(cfg["data"].get("zero_action_ratio", 0.0)),
        seed=seed,
        target_score_keys=cfg["data"].get("target_score_keys"),
    )

    train_indices, eval_indices = split_dataset_indices(
        len(dataset),
        float(cfg["data"].get("eval_ratio", 0.0)),
        seed,
    )
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices) if eval_indices else None
    if len(train_dataset) == 0:
        raise ValueError("train dataset is empty. Increase pair count or reduce eval_ratio.")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    model_name = model_cfg["name_or_path"]
    torch_dtype = torch_dtype_from_name(model_cfg.get("dtype", model_cfg.get("torch_dtype", "bfloat16")))
    attn_implementation = model_cfg.get("attn_implementation", None)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model_load_kwargs = {
        "dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_load_kwargs,
    )

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    run_root = Path(train_cfg["output_root"])
    run_name = str(train_cfg["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{timestamp}_{run_name}"
    ckpt_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    deepspeed_cfg = train_cfg.get("deepspeed_config")
    if deepspeed_cfg:
        deepspeed_cfg = str(deepspeed_cfg)

    per_device_train_batch_size = int(train_cfg["per_device_train_batch_size"])
    per_device_eval_batch_size = int(train_cfg["per_device_eval_batch_size"])
    gradient_accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    num_train_epochs = float(train_cfg["num_train_epochs"])
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0 and "warmup_ratio" in train_cfg:
        warmup_steps = compute_warmup_steps(
            train_size=len(train_dataset),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            warmup_ratio=float(train_cfg["warmup_ratio"]),
        )

    training_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_steps=warmup_steps,
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_strategy="steps",
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        deepspeed=deepspeed_cfg,
    )
    eval_strategy_value = "steps" if eval_dataset is not None else "no"
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_arg_names:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    else:
        training_kwargs["eval_strategy"] = eval_strategy_value
    training_args = TrainingArguments(**training_kwargs)

    collator = QwenVLScoreCollator(
        processor=processor,
        max_length=int(train_cfg["max_length"]),
        target_score_keys=dataset.target_score_keys,
        action_frame=dataset.action_frame,
        rotation_representation=dataset.rotation_representation,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "final_model_dir": str(final_dir),
        "dataset_pairs_total": len(dataset),
        "dataset_pairs_train": len(train_dataset),
        "dataset_pairs_eval": 0 if eval_dataset is None else len(eval_dataset),
        "views_total": len(dataset.views),
        "zero_action_pairs_total": int(dataset.zero_action_pairs_count),
        "action_frame": dataset.action_frame,
        "rotation_representation": dataset.rotation_representation,
        "target_score_keys": dataset.target_score_keys,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


"""example usage:
python scripts/train.py --config configs/qwen25_vl_7b_full.yaml
"""
