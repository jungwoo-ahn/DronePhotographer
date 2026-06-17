"""Train the π0-style VLA ablation baseline (issue #22).

  python scripts/train_vla_policy.py --config configs/policy/vla_qwen3_2b.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    import torch.distributed as dist
    from transformers import AutoProcessor, Qwen3VLModel

    # torchrun sets WORLD_SIZE/LOCAL_RANK/RANK → DDP mode.
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if is_distributed:
        from datetime import timedelta

        torch.cuda.set_device(local_rank)
        # Long timeout: rank-0-only validation (slow sampler over the 2B model)
        # stalls the other ranks' next collective; the 10-min default trips a
        # watchdog NCCL timeout and aborts the job. 2h covers any val pass.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        cfg["trainer"]["device"] = f"cuda:{local_rank}"
    is_main = rank == 0

    dtype = getattr(torch, cfg["backbone"]["dtype"])
    repo = cfg["backbone"]["repo_id"]
    # Base VLM (no LM head): we only need its hidden states as context.
    # Rank 0 downloads first; the rest wait and hit the warm cache.
    if is_distributed and not is_main:
        dist.barrier()
    # The vision tower is the bottleneck: its windowed attention falls back to a
    # slow Python loop without flash-attn (~4s/image, GPU idle). flash_attention_2
    # routes it through the varlen kernel. Configurable; defaults to sdpa.
    attn_impl = cfg["backbone"].get("attn_implementation", "sdpa")
    backbone = Qwen3VLModel.from_pretrained(repo, torch_dtype=dtype, attn_implementation=attn_impl)
    # Cap the vision-token count: at native 480x720 Qwen3-VL emits a huge token
    # sequence (full-FT attention over it is minutes/step). max_pixels bounds it
    # — framing is a global-composition task, so a few hundred tokens suffice.
    proc_kwargs = {}
    if cfg["backbone"].get("max_pixels"):
        proc_kwargs["max_pixels"] = int(cfg["backbone"]["max_pixels"])
    if cfg["backbone"].get("min_pixels"):
        proc_kwargs["min_pixels"] = int(cfg["backbone"]["min_pixels"])
    processor = AutoProcessor.from_pretrained(repo, **proc_kwargs)
    if is_distributed and is_main:
        dist.barrier()
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
    if is_main:
        print(f"dataset size: {len(dataset)}" + (f" | val: {len(val_dataset)}" if val_dataset is not None else ""))

    collate = VLACollate(processor, cfg["backbone"].get("prompt", "Describe the camera framing of the subject."))
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
        early_stop_patience=int(cfg["trainer"].get("early_stop_patience", 0)),
    )
    trainer = VLATrainer(policy, tcfg)
    if is_main:
        (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader, val_loader)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
