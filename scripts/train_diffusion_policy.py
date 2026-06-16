"""Train the Diffusion Policy ablation baseline (issue #22).

  python scripts/train_diffusion_policy.py --config configs/policy/diffusion_policy_dinov2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.policy.diffusion_policy.dataset import DPCollate, DiffusionPolicyDataset
from src.policy.diffusion_policy.model import DiffusionPolicy
from src.policy.diffusion_policy.trainer import DPTrainer, DPTrainerConfig


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
    from transformers import AutoImageProcessor, AutoModel

    dtype = getattr(torch, cfg["backbone"]["dtype"])
    repo = cfg["backbone"]["repo_id"]
    backbone = AutoModel.from_pretrained(repo, torch_dtype=dtype)
    processor = AutoImageProcessor.from_pretrained(repo)

    policy = DiffusionPolicy(
        backbone,
        goal_dim=len(cfg["data"]["goal_score_keys"]),
        goal_embed_dim=cfg["model"].get("goal_embed_dim", 128),
        chunk_size=cfg["data"]["chunk_size"],
        down_dims=tuple(cfg["model"].get("down_dims", [128, 256, 512])),
        diffusion_step_embed_dim=cfg["model"].get("diffusion_step_embed_dim", 128),
        num_train_timesteps=cfg["model"].get("num_train_timesteps", 100),
        beta_schedule=cfg["model"].get("beta_schedule", "squaredcos_cap_v2"),
        freeze_backbone=cfg["backbone"].get("freeze_backbone", True),
        processor=processor,
    )

    # Held-out validation split — same scene-level manifest as Cosmos/VLA
    # (configs/policy/val_scenes.txt), so the baselines are directly comparable.
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
    dataset = DiffusionPolicyDataset(
        cfg["data"]["annotation_roots"], stride=cfg["data"].get("stride", 1),
        max_samples=cfg["data"].get("max_samples"),
        goal_sampling=cfg["data"].get("goal_sampling", "uniform_future"),
        split="train", **common,
    )
    val_dataset = None
    if val_pair_stride > 0 or val_names:
        val_dataset = DiffusionPolicyDataset(
            cfg["data"]["annotation_roots"], stride=cfg["data"].get("val_stride", 4),
            goal_sampling="end", split="val", **common,
        )
        val_max = int(cfg["data"].get("val_max_samples", 64))
        if val_max and len(val_dataset) > val_max:
            from torch.utils.data import Subset
            idx = sorted({round(i * (len(val_dataset) - 1) / (val_max - 1)) for i in range(val_max)})
            val_dataset = Subset(val_dataset, idx)
    print(f"dataset size: {len(dataset)}" + (f" | val: {len(val_dataset)}" if val_dataset is not None else ""))

    collate = DPCollate(processor)
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

    tcfg = DPTrainerConfig(
        output_root=Path(cfg["trainer"]["output_root"]),
        run_name=cfg["trainer"]["run_name"],
        max_iter=cfg["trainer"]["max_iter"],
        batch_size=cfg["trainer"]["batch_size"],
        grad_accum=cfg["trainer"].get("grad_accum", 1),
        learning_rate=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"].get("weight_decay", 1e-6),
        warmup_iter=cfg["trainer"]["warmup_iter"],
        save_iter=cfg["trainer"]["save_iter"],
        log_iter=cfg["trainer"]["log_iter"],
        seed=cfg["trainer"]["seed"],
        device=cfg["trainer"]["device"],
        dtype=cfg["trainer"]["dtype"],
        val_iter=int(cfg["data"].get("val_iter", 0)),
        val_sample_steps=int(cfg["data"].get("val_sample_steps", 16)),
    )
    trainer = DPTrainer(policy, tcfg)
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader, val_loader)


if __name__ == "__main__":
    main()
