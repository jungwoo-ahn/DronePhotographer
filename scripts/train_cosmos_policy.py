"""End-to-end training entry point for the Cosmos-on-drone policy.

Reads a YAML config, builds the dataset/model/trainer, runs `.fit()`.

Usage:
  PYTHONPATH=. python scripts/train_cosmos_policy.py \
      --config configs/policy/cosmos_2b.yaml

  # smoke (50 iter, 32 samples):
  PYTHONPATH=. python scripts/train_cosmos_policy.py \
      --config configs/policy/cosmos_2b.yaml --debug

TensorBoard:
  tensorboard --logdir runs/<timestamp>_<run_name>/tb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, random_split

# Make `from src.policy.* import ...` work whether or not the repo root is on
# PYTHONPATH (e.g. ad-hoc launches from inside scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
        cfg["trainer"]["log_iter"] = 5
        cfg["trainer"]["val_iter"] = 25
        cfg["trainer"]["max_val_batches"] = 4
        cfg["data"]["max_samples"] = 32
    if args.max_iter is not None:
        cfg["trainer"]["max_iter"] = args.max_iter
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples

    from diffusers import DiffusionPipeline

    backbone_dtype = getattr(torch, cfg["backbone"]["dtype"])
    backbone_device = cfg["trainer"]["device"]

    # `device_map='cuda'` makes diffusers wrap the model with accelerate hooks
    # that decorate forward() with torch.no_grad(). Even though our backbone
    # weights are frozen, we DO need autograd through the cross-attention path
    # so the conditioner can learn. Load on CPU, then move manually.
    pipe = DiffusionPipeline.from_pretrained(
        cfg["backbone"]["repo_id"],
        torch_dtype=backbone_dtype,
    )
    pipe.transformer.to(backbone_device, dtype=backbone_dtype)
    pipe.vae.to(backbone_device, dtype=backbone_dtype)

    # Side-effect from `cosmos_guardrail`'s safety checker: loading the
    # pipeline flips the global autograd mode off (likely a stray
    # `torch.set_grad_enabled(False)` for inference). Re-enable it or every
    # `.backward()` we run from here will hit "element 0 does not require grad".
    if not torch.is_grad_enabled():
        torch.set_grad_enabled(True)
        print("[trainer] note: re-enabled global autograd (was disabled by pipeline load)")

    vae = CosmosVAEWrapper(pipe.vae)
    loss_cfg = cfg.get("loss", {})
    edm_cfg_dict = cfg.get("edm", {})
    edm_cfg = EDMConfig(**{k: v for k, v in edm_cfg_dict.items() if k in EDMConfig.__dataclass_fields__})
    policy = CosmosWorldActionPolicy(
        pipe.transformer,
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
    n_all = len(dataset)
    print(f"[data] dataset size: {n_all}")

    # ---- train / val split ---------------------------------------------------
    val_cfg = cfg.get("val", {})
    val_fraction = float(val_cfg.get("fraction", 0.0))
    val_seed = int(val_cfg.get("seed", 0))
    val_loader = None
    if 0.0 < val_fraction < 1.0 and n_all > 1:
        n_val = max(1, int(round(n_all * val_fraction)))
        n_train = n_all - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(val_seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["trainer"]["batch_size"],
            num_workers=cfg["dataloader"]["num_workers"],
            pin_memory=cfg["dataloader"]["pin_memory"],
            shuffle=cfg["dataloader"]["shuffle"],
            drop_last=cfg["dataloader"]["drop_last"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["trainer"]["batch_size"],
            num_workers=max(1, cfg["dataloader"]["num_workers"] // 2),
            pin_memory=cfg["dataloader"]["pin_memory"],
            shuffle=False,
            drop_last=False,
        )
        print(f"[data] split: train={n_train}  val={n_val} (fraction={val_fraction})")
    else:
        train_loader = DataLoader(
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
        val_iter=int(cfg["trainer"].get("val_iter", 0)),
        max_val_batches=int(cfg["trainer"].get("max_val_batches", 50)),
        keep_last_n=int(cfg["trainer"].get("keep_last_n", 3)),
        best_metric=str(cfg["trainer"].get("best_metric", "total")),
        seed=cfg["trainer"]["seed"],
        device=cfg["trainer"]["device"],
        dtype=cfg["trainer"]["dtype"],
    )
    trainer = CosmosPolicyTrainer(policy, vae, trainer_cfg)
    (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"[trainer] run_dir = {trainer.run_dir}")
    trainer.fit(train_loader, dataloader_val=val_loader)


if __name__ == "__main__":
    main()
