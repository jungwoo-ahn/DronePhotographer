"""End-to-end training entry point for the Cosmos-on-drone policy.

Reads a YAML config, builds the dataset/model/trainer, runs `.fit()`.

Usage:
  # single GPU
  python scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml

  # multi-GPU (DDP) — one process per GPU; per-GPU batch_size comes from the
  # config, so the effective batch is batch_size * grad_accum * nproc.
  torchrun --standalone --nproc_per_node=2 scripts/train_cosmos_policy.py \
      --config configs/policy/cosmos_2b.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.policy.cosmos.dataset import CosmosDroneDataset
from src.policy.cosmos.edm import FlowConfig
from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.trainer import CosmosPolicyTrainer, TrainerConfig
from src.policy.cosmos.vae import CosmosVAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--max_iter", type=int, default=None, help="CLI override")
    p.add_argument("--max_samples", type=int, default=None, help="CLI override")
    p.add_argument("--warmup_iter", type=int, default=None, help="CLI override (full config's 1000 assumes a 50k-iter run)")
    p.add_argument("--log_iter", type=int, default=None, help="CLI override (it/s log cadence)")
    p.add_argument("--save_iter", type=int, default=None, help="CLI override (checkpoint cadence)")
    p.add_argument("--resume", type=str, default=None,
                   help="resume from a run dir or ckpt .pt (restores optimizer/scheduler/iteration)")
    p.add_argument("--debug", action="store_true", help="reduce iters + samples for a smoke run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    if args.debug:
        cfg["trainer"]["max_iter"] = 50
        cfg["trainer"]["save_iter"] = 50
        cfg["trainer"]["warmup_iter"] = 10
        cfg["data"]["max_samples"] = 32
    if args.max_iter is not None:
        cfg["trainer"]["max_iter"] = args.max_iter
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.warmup_iter is not None:
        cfg["trainer"]["warmup_iter"] = args.warmup_iter
    if args.log_iter is not None:
        cfg["trainer"]["log_iter"] = args.log_iter
    if args.save_iter is not None:
        cfg["trainer"]["save_iter"] = args.save_iter

    import torch
    import torch.distributed as dist
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel

    # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; their presence selects DDP mode.
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        cfg["trainer"]["device"] = f"cuda:{local_rank}"
    is_main = rank == 0

    # Load only the two components we use. The text encoder (Qwen2.5-VL) is
    # bypassed entirely by our shot-profile conditioner, so we never download it.
    # The diffusers-format weights live on a branch of the NVIDIA repo.
    # Rank 0 downloads first; the rest wait and hit the warm cache.
    if is_distributed and not is_main:
        dist.barrier()
    dtype = getattr(torch, cfg["backbone"]["dtype"])
    revision = cfg["backbone"].get("revision", "diffusers/base/post-trained")
    transformer = CosmosTransformer3DModel.from_pretrained(
        cfg["backbone"]["repo_id"], subfolder="transformer", revision=revision, torch_dtype=dtype,
    )
    # Full finetune needs activation recomputation to fit 2B params + grads +
    # Adam moments on a 40GB card. No effect when the backbone is frozen.
    if cfg["backbone"].get("gradient_checkpointing", not cfg["backbone"]["freeze_backbone"]):
        transformer.enable_gradient_checkpointing()
    raw_vae = AutoencoderKLWan.from_pretrained(
        cfg["backbone"]["repo_id"], subfolder="vae", revision=revision, torch_dtype=dtype,
    )
    if is_distributed and is_main:
        dist.barrier()

    vae = CosmosVAEWrapper(raw_vae)

    # The 2.5 transformer projects text features internally (crossattn_proj):
    # its INPUT dim is the text encoder's stacked hidden states (Qwen2.5-VL,
    # 3584 x 28 layers = 100352), not config.text_embed_dim (the post-projection
    # 1024). Our goal tokens enter pre-projection, so size the conditioner to
    # the projection's true in_features.
    from torch import nn as _nn

    crossattn_dim = 1024
    proj = getattr(transformer, "crossattn_proj", None)
    if proj is not None:
        crossattn_dim = next(m.in_features for m in proj.modules() if isinstance(m, _nn.Linear))
    if is_main:
        print(f"conditioner cross-attention dim: {crossattn_dim}")

    loss_cfg = cfg.get("loss", {})
    flow_cfg_dict = cfg.get("flow", {})
    flow_cfg = FlowConfig(**{k: v for k, v in flow_cfg_dict.items() if k in FlowConfig.__dataclass_fields__})
    policy = CosmosWorldActionPolicy(
        transformer,
        crossattn_dim=crossattn_dim,
        goal_dim=len(cfg["data"]["goal_score_keys"]),
        n_goal_tokens=cfg["backbone"]["n_goal_tokens"],
        freeze_backbone=cfg["backbone"]["freeze_backbone"],
        anchor_path=cfg.get("conditioner", {}).get("anchor_path"),
        chunk_size=cfg["data"]["chunk_size"],
        lambda_world=float(loss_cfg.get("lambda_world", 1.0)),
        lambda_action=float(loss_cfg.get("lambda_action", 1.0)),
        lambda_value=float(loss_cfg.get("lambda_value", 1.0)),
        goal_adaln=cfg["backbone"].get("goal_adaln", False),
        flow_config=flow_cfg,
    )

    val_pair_stride = int(cfg["data"].get("val_pair_stride", 0))
    val_split_level = cfg["data"].get("val_split_level", "pair")
    # Frozen manifest: a YAML list of unit names, or a path to a txt file
    # (one name per line, '#' comments). Overrides the hash split — the val
    # set is pinned and all future data arrivals are train.
    val_names = cfg["data"].get("val_names")
    if isinstance(val_names, str):
        val_names = [
            ln.strip() for ln in Path(val_names).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    dataset = CosmosDroneDataset(
        cfg["data"]["annotation_roots"],
        goal_score_keys=cfg["data"]["goal_score_keys"],
        chunk_size=cfg["data"]["chunk_size"],
        stride=cfg["data"].get("stride", 1),
        max_samples=cfg["data"].get("max_samples"),
        target_resolution=tuple(cfg["data"]["target_resolution"]),
        goal_sampling=cfg["data"].get("goal_sampling", "uniform_future"),
        val_pair_stride=val_pair_stride,
        val_split_level=val_split_level,
        val_names=val_names,
        split="train",
        augment_reverse=cfg["data"].get("augment_reverse", False),
    )
    val_dataset = None
    if val_pair_stride > 0 or val_names:
        # Held-out trajectory pairs. goal_sampling="end" pins val goals
        # deterministically so the metrics don't jitter with the HER draw.
        val_dataset = CosmosDroneDataset(
            cfg["data"]["annotation_roots"],
            goal_score_keys=cfg["data"]["goal_score_keys"],
            chunk_size=cfg["data"]["chunk_size"],
            stride=cfg["data"].get("val_stride", 4),
            target_resolution=tuple(cfg["data"]["target_resolution"]),
            goal_sampling="end",
            val_pair_stride=val_pair_stride,
            val_split_level=val_split_level,
            val_names=val_names,
            split="val",
            augment_reverse=cfg["data"].get("augment_reverse", False),
        )
        # Cap val cost by subsampling EVENLY across the val set (a head-truncation
        # via max_samples would take all windows from the first scene only).
        val_max = int(cfg["data"].get("val_max_samples", 64))
        if val_max and len(val_dataset) > val_max:
            from torch.utils.data import Subset

            idx = [round(i * (len(val_dataset) - 1) / (val_max - 1)) for i in range(val_max)]
            val_dataset = Subset(val_dataset, sorted(set(idx)))
    if is_main:
        print(f"dataset size: {len(dataset)}" + (f" | val: {len(val_dataset)}" if val_dataset is not None else ""))

    # Under DDP each rank sees a disjoint shard; the sampler owns the shuffle
    # (trainer calls set_epoch each pass).
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
        sampler=sampler,
    )
    # Validation runs on rank 0 only — plain loader, no sharding.
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg["trainer"]["batch_size"], num_workers=2)
        if val_dataset is not None and is_main else None
    )

    trainer_cfg = TrainerConfig(
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
        val_iter=int(cfg["trainer"].get("val_iter", 0)),
        val_sample_steps=int(cfg["trainer"].get("val_sample_steps", 8)),
        resume_from=(args.resume if args.resume is not None else cfg["trainer"].get("resume_from")),
    )
    trainer = CosmosPolicyTrainer(policy, vae, trainer_cfg)
    if is_main:
        (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader, val_loader)
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
