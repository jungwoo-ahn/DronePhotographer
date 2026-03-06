import argparse
import os
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import DronePairDataset
from src.models.q_model import VisionQNetwork
from src.models.vit_q_model import VisionTransformerQNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_configs/q_model.yaml")
    return parser.parse_args()


def _cfg_get(cfg: dict, keys: list[str], default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_root = _cfg_get(cfg, ["data", "data_root"], "outputs/DogWalk_260204_131211")
    distance_threshold = float(_cfg_get(cfg, ["data", "distance_threshold"], 1.5))
    batch_size = int(_cfg_get(cfg, ["data", "batch_size"], 16))
    num_workers = int(_cfg_get(cfg, ["data", "num_workers"], 0))
    image_size = _cfg_get(cfg, ["data", "image_size"], None)
    pair_cache_path = _cfg_get(cfg, ["data", "pair_cache_path"], None)

    max_steps = int(_cfg_get(cfg, ["train", "max_steps"], 2000))
    lr = float(_cfg_get(cfg, ["train", "lr"], 1e-4))
    device_cfg = _cfg_get(cfg, ["train", "device"], "auto")
    device = "cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else device_cfg

    run_root = _cfg_get(cfg, ["run", "run_root"], "runs")
    run_name = _cfg_get(cfg, ["run", "run_name"], "q_model")
    save_every_steps = int(_cfg_get(cfg, ["run", "save_every_steps"], 200))

    model_name = _cfg_get(cfg, ["model", "name"], "cnn")
    action_dim = int(_cfg_get(cfg, ["model", "action_dim"], 6))
    num_outputs = int(_cfg_get(cfg, ["model", "num_outputs"], 5))

    if image_size:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()

    dataset = DronePairDataset(
        root=data_root,
        distance_threshold=distance_threshold,
        image_transform=transform,
        pair_cache_path=pair_cache_path,
    )
    print(
        f"dataset: images={len(dataset.images)} pairs={len(dataset)} distance_threshold={distance_threshold}"
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, f"{timestamp}_{run_name}")
    ckpt_dir = os.path.join(run_dir, "ckpts")
    tb_dir = os.path.join(run_dir, "tensorboard")
    summary_path = os.path.join(run_dir, "summary.txt")
    config_copy_path = os.path.join(run_dir, "config.yaml")
    render_info_src = os.path.join(data_root, "run_info.json")
    render_info_dst = os.path.join(run_dir, "render_run_info.json")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    with open(config_copy_path, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(cfg, sort_keys=False))

    if os.path.exists(render_info_src):
        with open(render_info_src, "r", encoding="utf-8") as f:
            render_info = f.read()
        with open(render_info_dst, "w", encoding="utf-8") as f:
            f.write(render_info)

    steps_per_epoch = max(1, len(loader))

    if model_name == "cnn":
        feature_dim = int(_cfg_get(cfg, ["model", "cnn", "feature_dim"], 256))
        hidden_dim = int(_cfg_get(cfg, ["model", "cnn", "hidden_dim"], 256))
        model = VisionQNetwork(
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_outputs=num_outputs,
        ).to(device)
    elif model_name == "vit":
        patch_size = int(_cfg_get(cfg, ["model", "vit", "patch_size"], 16))
        embed_dim = int(_cfg_get(cfg, ["model", "vit", "embed_dim"], 256))
        depth = int(_cfg_get(cfg, ["model", "vit", "depth"], 4))
        num_heads = int(_cfg_get(cfg, ["model", "vit", "num_heads"], 8))
        mlp_dim = int(_cfg_get(cfg, ["model", "vit", "mlp_dim"], 512))
        dropout = float(_cfg_get(cfg, ["model", "vit", "dropout"], 0.0))
        model = VisionTransformerQNetwork(
            action_dim=action_dim,
            num_outputs=num_outputs,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        ).to(device)
    elif model_name == "dino_cross":
        from src.models.dino_cross_q_model import DinoCrossAttentionQNetwork

        dino_model_name = _cfg_get(
            cfg,
            ["model", "dino_cross", "dino_model_name"],
            "facebook/dinov2-base",
        )
        image_size_model = int(_cfg_get(cfg, ["model", "dino_cross", "image_size"], 224))
        embed_dim = int(_cfg_get(cfg, ["model", "dino_cross", "embed_dim"], 256))
        depth = int(_cfg_get(cfg, ["model", "dino_cross", "depth"], 3))
        num_heads = int(_cfg_get(cfg, ["model", "dino_cross", "num_heads"], 8))
        mlp_dim = int(_cfg_get(cfg, ["model", "dino_cross", "mlp_dim"], 512))
        dropout = float(_cfg_get(cfg, ["model", "dino_cross", "dropout"], 0.0))
        head_hidden_dim = int(_cfg_get(cfg, ["model", "dino_cross", "head_hidden_dim"], 256))

        model = DinoCrossAttentionQNetwork(
            action_dim=action_dim,
            num_outputs=num_outputs,
            dino_model_name=dino_model_name,
            image_size=image_size_model,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"unknown model name: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    writer = SummaryWriter(tb_dir)

    model.train()
    global_step = 0
    epoch = 0
    total_loss = 0.0
    total_score_losses = [0.0 for _ in range(num_outputs)]
    total_batches = 0
    last_step_loss = None
    last_epoch_loss = None

    while global_step < max_steps:
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}", total=steps_per_epoch)
        for images, delta_pos, delta_rot, targets in pbar:
            if global_step >= max_steps:
                break

            global_step += 1
            images = images.to(device)
            delta_pos = delta_pos.to(device)
            delta_rot = delta_rot.to(device)
            targets = targets.to(device)

            actions = torch.cat([delta_pos, delta_rot], dim=1)
            preds = model(images, actions)
            loss = loss_fn(preds, targets)
            score_losses = ((preds - targets) ** 2).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for idx in range(num_outputs):
                total_score_losses[idx] += score_losses[idx].item()
            total_batches += 1
            last_step_loss = loss.item()

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            for idx in range(num_outputs):
                writer.add_scalar(
                    f"train/loss_step_score{idx}",
                    score_losses[idx].item(),
                    global_step,
                )
            pbar.set_postfix(step=global_step, loss=loss.item())

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                step_path = os.path.join(ckpt_dir, f"q_model_step{global_step}.pt")
                torch.save(model.state_dict(), step_path)

        epoch += 1
        avg_loss = total_loss / max(1, total_batches)
        last_epoch_loss = avg_loss
        print(f"epoch {epoch} | loss {avg_loss:.6f}")
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        for idx in range(num_outputs):
            writer.add_scalar(
                f"train/loss_epoch_score{idx}",
                total_score_losses[idx] / max(1, total_batches),
                epoch,
            )

        epoch_path = os.path.join(ckpt_dir, f"q_model_epoch{epoch}.pt")
        torch.save(model.state_dict(), epoch_path)

        total_loss = 0.0
        total_score_losses = [0.0 for _ in range(num_outputs)]
        total_batches = 0

    final_path = os.path.join(ckpt_dir, "q_model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"saved: {final_path}")
    writer.close()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"config_path: {os.path.abspath(args.config)}\n")
        f.write(f"data_root: {data_root}\n")
        f.write(f"distance_threshold: {distance_threshold}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"max_steps: {max_steps}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"num_workers: {num_workers}\n")
        f.write(f"device: {device}\n")
        f.write(f"save_every_steps: {save_every_steps}\n")
        f.write(f"steps_per_epoch: {steps_per_epoch}\n")
        f.write(f"ckpt_dir: {ckpt_dir}\n")
        f.write(f"tensorboard_dir: {tb_dir}\n")
        f.write(f"final_ckpt: {final_path}\n")
        f.write(f"last_step_loss: {last_step_loss}\n")
        f.write(f"last_epoch_loss: {last_epoch_loss}\n")
        f.write("\ndataset_stats:\n")
        stats = getattr(dataset, "stats", None)
        if isinstance(stats, dict):
            f.write(yaml.safe_dump(stats, sort_keys=False))
        else:
            f.write("null\n")
        f.write("\nconfig:\n")
        f.write(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
    
"""
CUDA_VISIBLE_DEVICES=3 python train.py --config train_configs/dino_cross_model.yaml
"""
