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
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.collator import QwenVLScoreCollator
from src.vlm_qwen25.dataset import DroneActionScoreDataset
from src.vlm_qwen25.schema import parse_scores_from_text
from src.scoring.evaluator import RULE_BASED_SCORE_KEYS, CAMERA_3D_SCORE_KEYS


class PredictionLoggingCallback(TrainerCallback):
    """Log model predictions vs GT to TensorBoard during eval."""

    def __init__(self, eval_dataset, processor, score_keys, num_samples=32, max_new_tokens=256):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.score_keys = list(score_keys)
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        # Group keys
        self.bbox_keys = [k for k in self.score_keys if k in RULE_BASED_SCORE_KEYS]
        self.camera3d_keys = [k for k in self.score_keys if k in CAMERA_3D_SCORE_KEYS]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        tb_writer = None
        for cb_obj in kwargs.get("callbacks", []):
            if hasattr(cb_obj, "tb_writer"):
                tb_writer = cb_obj.tb_writer
                break
        if tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_writer = SummaryWriter(log_dir=args.logging_dir)
            except Exception:
                return

        from src.vlm_qwen25.prompt import build_user_prompt

        model.eval()
        device = next(model.parameters()).device
        indices = list(range(min(self.num_samples, len(self.eval_dataset))))

        all_errors = {k: [] for k in self.score_keys}
        parse_failures = 0
        last_pred = None
        last_gt = None

        for idx in indices:
            sample = self.eval_dataset[idx]
            image = sample["image"]
            action_text = sample["action_text"]
            gt_scores = sample["target_scores"]

            user_prompt = build_user_prompt(
                action_text=action_text,
                target_score_keys=self.score_keys,
            )
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ]}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt_text], images=[image],
                return_tensors="pt", padding=True,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_len:]
            pred_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            pred_scores = parse_scores_from_text(pred_text, self.score_keys)

            if pred_scores is None:
                parse_failures += 1
                continue

            last_pred = pred_scores
            last_gt = gt_scores
            for k in self.score_keys:
                if k in pred_scores and k in gt_scores:
                    all_errors[k].append(abs(pred_scores[k] - gt_scores[k]))

        step = state.global_step

        # Per-key MAE
        for k in self.score_keys:
            errs = all_errors[k]
            if errs:
                tb_writer.add_scalar(f"pred_vs_gt/{k}_mae", sum(errs) / len(errs), step)

        # Group MAE: bbox vs camera_3d vs total
        def _group_mae(keys):
            errs = []
            for k in keys:
                errs.extend(all_errors.get(k, []))
            return sum(errs) / len(errs) if errs else None

        bbox_mae = _group_mae(self.bbox_keys)
        cam3d_mae = _group_mae(self.camera3d_keys)
        total_mae = _group_mae(self.score_keys)

        if bbox_mae is not None:
            tb_writer.add_scalar("pred_vs_gt/bbox_group_mae", bbox_mae, step)
        if cam3d_mae is not None:
            tb_writer.add_scalar("pred_vs_gt/camera3d_group_mae", cam3d_mae, step)
        if total_mae is not None:
            tb_writer.add_scalar("pred_vs_gt/total_mae", total_mae, step)

        if indices:
            tb_writer.add_scalar(
                "pred_vs_gt/parse_failure_rate",
                parse_failures / len(indices),
                step,
            )

        # Log one sample detail as text
        if last_pred is not None and last_gt is not None:
            detail = "| key | pred | gt | err |\n|---|---|---|---|\n"
            for k in self.score_keys:
                p = last_pred.get(k, float("nan"))
                g = last_gt.get(k, float("nan"))
                detail += f"| {k} | {p:.4f} | {g:.4f} | {abs(p-g):.4f} |\n"
            tb_writer.add_text("pred_vs_gt/sample", detail, step)

        tb_writer.flush()


class ScoreRegressionTrainer(Trainer):
    """Trainer with auxiliary regression loss on predicted scores."""

    def __init__(self, *args, score_keys=None, regression_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_keys = score_keys or []
        self.regression_weight = float(regression_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gt_scores = inputs.pop("gt_score_values", None)

        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss

        if (
            gt_scores is not None
            and hasattr(model, "score_head")
            and outputs.hidden_states is not None
        ):
            hidden = outputs.hidden_states[-1]  # [B, seq, hidden]
            labels = inputs["labels"]
            # prompt end = first position where labels != -100, minus 1
            prompt_ends = (labels != -100).long().argmax(dim=1) - 1
            prompt_ends = prompt_ends.clamp(min=0)

            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            prompt_hidden = hidden[batch_idx, prompt_ends]  # [B, hidden]

            pred_scores = model.score_head(prompt_hidden)  # [B, num_keys]
            gt_scores = gt_scores.to(pred_scores.device, dtype=pred_scores.dtype)
            reg_loss = torch.nn.functional.l1_loss(pred_scores, gt_scores)

            loss = lm_loss + self.regression_weight * reg_loss
        else:
            loss = lm_loss

        return (loss, outputs) if return_outputs else loss


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

    # Score prediction head for auxiliary regression loss
    regression_weight = float(train_cfg.get("regression_weight", 0.0))
    if regression_weight > 0:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config.text_config, "hidden_size", 1536)
        num_score_keys = len(cfg["data"].get("target_score_keys", []))
        model.score_head = torch.nn.Linear(hidden_size, num_score_keys).to(torch_dtype)
        print(f"Score head added: {hidden_size} → {num_score_keys} (regression_weight={regression_weight})", flush=True)

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
        logging_dir=str(run_dir / "tb_logs"),
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

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(PredictionLoggingCallback(
            eval_dataset=eval_dataset,
            processor=processor,
            score_keys=dataset.target_score_keys,
            num_samples=min(32, len(eval_dataset)),
        ))

    if regression_weight > 0:
        trainer = ScoreRegressionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
            score_keys=dataset.target_score_keys,
            regression_weight=regression_weight,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

    # Log hyperparameters to TensorBoard for cross-run comparison
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
    hparams = {
        "model": model_name,
        "num_views": len(dataset.views),
        "num_train_pairs": len(train_dataset),
        "num_eval_pairs": 0 if eval_dataset is None else len(eval_dataset),
        "num_score_keys": len(dataset.target_score_keys),
        "effective_batch_size": effective_batch_size,
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": num_train_epochs,
        "total_steps": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "distance_threshold": float(cfg["data"]["distance_threshold"]),
        "action_frame": action_frame,
        "rotation_representation": rotation_representation,
    }
    if trainer.is_world_process_zero():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(ckpt_dir / "runs"))
            # Log hparams as text table
            hparams_text = "\n".join(f"| {k} | {v} |" for k, v in hparams.items())
            tb_writer.add_text("hparams", f"| key | value |\n|---|---|\n{hparams_text}", 0)
            for k, v in hparams.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f"hparams/{k}", v, 0)
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # Collect final training metrics
    final_metrics = {}
    if trainer.state.log_history:
        for entry in trainer.state.log_history:
            if "loss" in entry:
                final_metrics["final_train_loss"] = entry["loss"]
            if "eval_loss" in entry:
                final_metrics["final_eval_loss"] = entry["eval_loss"]

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "final_model_dir": str(final_dir),
        "config_path": str(cfg_path),
        "dataset_pairs_total": len(dataset),
        "dataset_pairs_train": len(train_dataset),
        "dataset_pairs_eval": 0 if eval_dataset is None else len(eval_dataset),
        "views_total": len(dataset.views),
        "zero_action_pairs_total": int(dataset.zero_action_pairs_count),
        "action_frame": dataset.action_frame,
        "rotation_representation": dataset.rotation_representation,
        "target_score_keys": dataset.target_score_keys,
        "hparams": hparams,
        "final_metrics": final_metrics,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


"""example usage:
python scripts/train.py --config configs/qwen25_vl_7b_full.yaml
"""
