"""Train the π0-style VLA ablation baseline (issue #22).

  python scripts/train_vla_policy.py --config configs/policy/vla_qwen3_2b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.policy.common.flow import FlowConfig
from src.policy.vla.dataset import VLADroneDataset
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

    dataset = VLADroneDataset(
        cfg["data"]["annotation_roots"],
        goal_score_keys=cfg["data"]["goal_score_keys"],
        chunk_size=cfg["data"]["chunk_size"],
        stride=cfg["data"].get("stride", 1),
        max_samples=cfg["data"].get("max_samples"),
        target_resolution=tuple(cfg["data"]["target_resolution"]),
    )
    print(f"dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        shuffle=cfg["dataloader"]["shuffle"],
        drop_last=cfg["dataloader"]["drop_last"],
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
    )
    trainer = VLATrainer(policy, tcfg)
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader)


if __name__ == "__main__":
    main()
