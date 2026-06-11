"""End-to-end training entry point for the Cosmos-on-drone policy.

Reads a YAML config, builds the dataset/model/trainer, runs `.fit()`.

Usage:
  python scripts/train_cosmos_policy.py --config configs/policy/cosmos_2b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.policy.cosmos.dataset import CosmosDroneDataset
from src.policy.cosmos.edm import EDMConfig
from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.trainer import CosmosPolicyTrainer, TrainerConfig
from src.policy.cosmos.vae import CosmosVAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--max_iter", type=int, default=None, help="CLI override")
    p.add_argument("--max_samples", type=int, default=None, help="CLI override")
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

    import torch
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel

    # Load only the two components we use. The text encoder (Qwen2.5-VL) is
    # bypassed entirely by our shot-profile conditioner, so we never download it.
    # The diffusers-format weights live on a branch of the NVIDIA repo.
    dtype = getattr(torch, cfg["backbone"]["dtype"])
    revision = cfg["backbone"].get("revision", "diffusers/base/post-trained")
    transformer = CosmosTransformer3DModel.from_pretrained(
        cfg["backbone"]["repo_id"], subfolder="transformer", revision=revision, torch_dtype=dtype,
    )
    raw_vae = AutoencoderKLWan.from_pretrained(
        cfg["backbone"]["repo_id"], subfolder="vae", revision=revision, torch_dtype=dtype,
    )

    vae = CosmosVAEWrapper(raw_vae)
    loss_cfg = cfg.get("loss", {})
    edm_cfg_dict = cfg.get("edm", {})
    edm_cfg = EDMConfig(**{k: v for k, v in edm_cfg_dict.items() if k in EDMConfig.__dataclass_fields__})
    policy = CosmosWorldActionPolicy(
        transformer,
        goal_dim=len(cfg["data"]["goal_score_keys"]),
        n_goal_tokens=cfg["backbone"]["n_goal_tokens"],
        freeze_backbone=cfg["backbone"]["freeze_backbone"],
        anchor_path=cfg.get("conditioner", {}).get("anchor_path"),
        chunk_size=cfg["data"]["chunk_size"],
        lambda_world=float(loss_cfg.get("lambda_world", 1.0)),
        lambda_action=float(loss_cfg.get("lambda_action", 1.0)),
        lambda_value=float(loss_cfg.get("lambda_value", 1.0)),
        edm_config=edm_cfg,
    )

    dataset = CosmosDroneDataset(
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
    )
    trainer = CosmosPolicyTrainer(policy, vae, trainer_cfg)
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    trainer.fit(loader)


if __name__ == "__main__":
    main()
