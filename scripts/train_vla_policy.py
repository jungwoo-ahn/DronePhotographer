"""Train the π0-style VLA ablation baseline (issue #22).

  python scripts/train_vla_policy.py --config configs/policy/vla_qwen3_2b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.policy.common.flow import FlowConfig
from src.policy.vla.dataset import VLACollate, VLADroneDataset
from src.policy.vla.model import VLAActionPolicy
from src.policy.vla.trainer import VLATrainer, VLATrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.max_iter is not None:
        cfg["trainer"]["max_iter"] = args.max_iter
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples

    import torch
    from transformers import AutoProcessor, Qwen3VLModel

    dtype = getattr(torch, cfg["backbone"]["dtype"])
    repo = cfg["backbone"]["repo_id"]
    # Base VLM (no LM head): we only need its hidden states as context.
    backbone = Qwen3VLModel.from_pretrained(repo, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(repo)
    if not cfg["backbone"]["freeze_backbone"] and cfg["backbone"].get("gradient_checkpointing", True):
        backbone.gradient_checkpointing_enable()

    flow_cfg = FlowConfig(**{k: v for k, v in cfg.get("flow", {}).items() if k in FlowConfig.__dataclass_fields__})
    policy = VLAActionPolicy(
        backbone,
        goal_dim=len(cfg["data"]["goal_score_keys"]),
        n_goal_tokens=cfg["backbone"]["n_goal_tokens"],
        chunk_size=cfg["data"]["chunk_size"],
        expert_dim=cfg["expert"]["dim"],
        expert_depth=cfg["expert"]["depth"],
        expert_heads=cfg["expert"]["heads"],
        freeze_backbone=cfg["backbone"]["freeze_backbone"],
        flow_config=flow_cfg,
        processor=processor,
        prompt=cfg["backbone"].get("prompt", "Describe the camera framing of the subject."),
    )

    # Held-out validation split — same scene-level manifest the Cosmos policy
    # uses (configs/policy/val_scenes.txt), so the two are directly comparable
    # and neither trains on the val scenes.
    val_split_level = cfg["data"].get("val_split_level", "scene")
    val_pair_stride = int(cfg["data"].get("val_pair_stride", 0))
    val_names = cfg["data"].get("val_names")
    if isinstance(val_names, str):
        val_names = [ln.strip() for ln in Path(val_names).read_text().splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]

    common = dict(
        goal_score_keys=cfg["data"]["goal_score_keys"],
        chunk_size=cfg["data"]["chunk_size"],
        target_resolution=tuple(cfg["data"]["target_resolution"]),
        val_pair_stride=val_pair_stride,
        val_split_level=val_split_level,
        val_names=val_names,
    )
    dataset = VLADroneDataset(
        cfg["data"]["annotation_roots"], stride=cfg["data"].get("stride", 1),
        max_samples=cfg["data"].get("max_samples"),
        goal_sampling=cfg["data"].get("goal_sampling", "uniform_future"),
        split="train", **common,
    )
    val_dataset = None
    if val_pair_stride > 0 or val_names:
        val_dataset = VLADroneDataset(
            cfg["data"]["annotation_roots"], stride=cfg["data"].get("val_stride", 4),
            goal_sampling="end", split="val", **common,
        )
        val_max = int(cfg["data"].get("val_max_samples", 64))
        if val_max and len(val_dataset) > val_max:
            from torch.utils.data import Subset
            idx = sorted({round(i * (len(val_dataset) - 1) / (val_max - 1)) for i in range(val_max)})
            val_dataset = Subset(val_dataset, idx)
    print(f"dataset size: {len(dataset)}" + (f" | val: {len(val_dataset)}" if val_dataset is not None else ""))

    collate = VLACollate(processor, cfg["backbone"].get("prompt", "Describe the camera framing of the subject."))
    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        shuffle=cfg["dataloader"]["shuffle"],
        drop_last=cfg["dataloader"]["drop_last"],
        collate_fn=collate,
        persistent_workers=cfg["dataloader"]["num_workers"] > 0,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg["trainer"]["batch_size"], num_workers=2, collate_fn=collate)
        if val_dataset is not None else None
    )

    tcfg = VLATrainerConfig(
        output_root=Path(cfg["trainer"]["output_root"]),
        run_name=cfg["trainer"]["run_name"],
        max_iter=cfg["trainer"]["max_iter"],
        batch_size=cfg["trainer"]["batch_size"],
        grad_accum=cfg["trainer"]["grad_accum"],
        learning_rate=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"]["weight_decay"],
        warmup_iter=cfg["trainer"]["warmup_iter"],
        save_iter=cfg["trainer"]["save_iter"],
        log_iter=cfg["trainer"]["log_iter"],
        seed=cfg["trainer"]["seed"],
        device=cfg["trainer"]["device"],
        dtype=cfg["trainer"]["dtype"],
        val_iter=int(cfg["data"].get("val_iter", 0)),
        val_sample_steps=int(cfg["data"].get("val_sample_steps", 10)),
    )
    trainer = VLATrainer(policy, tcfg)
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader, val_loader)


if __name__ == "__main__":
    main()
