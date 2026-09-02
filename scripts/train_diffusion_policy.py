"""Train the Diffusion Policy ablation baseline (issue #22).

  python scripts/train_diffusion_policy.py --config configs/policy/diffusion_policy_dinov2.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.policy.diffusion_policy.dataset import DPCollate, DiffusionPolicyDataset
from src.policy.diffusion_policy.model import DiffusionPolicy
from src.policy.diffusion_policy.trainer import DPTrainer, DPTrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--resume_from", type=str, default=None, help="checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.max_iter is not None:
        cfg["trainer"]["max_iter"] = args.max_iter
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples

    import torch
    import torch.distributed as dist
    from transformers import AutoImageProcessor, AutoModel

    # torchrun sets WORLD_SIZE/LOCAL_RANK/RANK → DDP mode.
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if is_distributed:
        from datetime import timedelta

        torch.cuda.set_device(local_rank)
        # Long timeout: rank-0-only validation + large (optimizer-state) checkpoint
        # writes to NFS stall the other rank's next collective; the 10-min default
        # can trip a watchdog NCCL timeout. 2h is ample.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        cfg["trainer"]["device"] = f"cuda:{local_rank}"
    is_main = rank == 0

    dtype = getattr(torch, cfg["backbone"]["dtype"])
    repo = cfg["backbone"]["repo_id"]
    # Rank 0 downloads first; the rest wait and hit the warm cache.
    if is_distributed and not is_main:
        dist.barrier()
    backbone = AutoModel.from_pretrained(repo, torch_dtype=dtype)
    processor = AutoImageProcessor.from_pretrained(repo)
    if is_distributed and is_main:
        dist.barrier()

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
    from src.policy.common.annotations import load_val_names
    val_names = load_val_names(cfg["data"].get("val_names"))

    common = dict(
        goal_score_keys=cfg["data"]["goal_score_keys"],
        chunk_size=cfg["data"]["chunk_size"],
        sampling_scheme=cfg["data"].get("sampling_scheme", "sliding_window"),
        offsets=cfg["data"].get("offsets", [8, 16, 24]),
        goal_start_max_per_pair=int(cfg["data"].get("goal_start_max_per_pair", 24)),
        goal_start_seed=int(cfg["data"].get("goal_start_seed", 0)),
        target_resolution=tuple(cfg["data"]["target_resolution"]),
        val_pair_stride=val_pair_stride,
        val_split_level=val_split_level,
        val_names=val_names,
        cache_dir=cfg["data"].get("cache_dir"),
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
    if is_main:
        print(f"dataset size: {len(dataset)}" + (f" | val: {len(val_dataset)}" if val_dataset is not None else ""))

    collate = DPCollate(processor)
    sampler = (
        DistributedSampler(dataset, shuffle=cfg["dataloader"]["shuffle"], drop_last=cfg["dataloader"]["drop_last"])
        if is_distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        shuffle=cfg["dataloader"]["shuffle"] if sampler is None else False,
        drop_last=cfg["dataloader"]["drop_last"],
        collate_fn=collate,
        persistent_workers=cfg["dataloader"]["num_workers"] > 0,
        sampler=sampler,
    )
    # Validation runs on rank 0 only — plain loader, no sharding.
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg["trainer"]["batch_size"], num_workers=2, collate_fn=collate)
        if val_dataset is not None and is_main else None
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
        resume_from=args.resume_from or cfg["trainer"].get("resume_from"),
        early_stop_patience=int(cfg["trainer"].get("early_stop_patience", 0)),
    )
    trainer = DPTrainer(policy, tcfg)
    if is_main:
        (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader, val_loader)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
