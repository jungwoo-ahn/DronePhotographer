"""Evaluate the Diffusion Policy baseline on a target shot profile.

Mirrors `scripts/eval_vla_policy.py` / `eval_cosmos_policy.py` so all three are
directly comparable: same v7 start frame, same goal YAML, same pose-proxy distance
metric and JSON output — only the model differs (frozen DINOv2 + DDPM action head,
no world model).

  python scripts/eval_diffusion_policy.py \
      --checkpoint runs/<ts>_diffusion_policy_dinov2/ckpt_best.pt \
      --start_annotation data/trajectories_full/<placement>/data.json \
      --target configs/policy/targets/centered_medium.yaml [--n_steps 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.policy.common.action_repr import apply_action_5d
from src.policy.common.annotations import iter_windows
from src.policy.common.goal_space import goal_keys, normalize_goal
from src.policy.common.reward import score_distance
from src.policy.diffusion_policy.dataset import _load_image_as_tensor
from src.policy.diffusion_policy.model import DiffusionPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--start_annotation", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path, help="YAML with `target:` {score_key: value}")
    p.add_argument("--repo_id", default="facebook/dinov2-large")
    p.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_steps", type=int, default=16, help="DDIM sampling steps")
    p.add_argument("--chunk_size", type=int, default=8, help="must match training")
    return p.parse_args()


def load_target_yaml(path: Path) -> dict[str, float]:
    cfg = yaml.safe_load(path.read_text())
    target = cfg.get("target", {})
    remap = {"center_x": "object_center_x", "center_y": "object_center_y"}
    return {remap.get(k, k): float(v) for k, v in target.items()}


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from transformers import AutoImageProcessor, AutoModel

    backbone = AutoModel.from_pretrained(args.repo_id, torch_dtype=dtype)
    processor = AutoImageProcessor.from_pretrained(args.repo_id)
    target_dict = load_target_yaml(args.target)
    policy = DiffusionPolicy(
        backbone, goal_dim=len(target_dict), chunk_size=args.chunk_size, processor=processor,
    ).to(device).eval()
    missing, unexpected = policy.load_state_dict(ckpt["policy_state"], strict=False)
    if missing or unexpected:
        print(f"warning: missing={len(missing)} unexpected={len(unexpected)} keys in checkpoint")

    target_keys = goal_keys(list(target_dict.keys()))
    target_vec = np.array([target_dict[k] for k in target_keys], dtype=np.float32)
    target_norm = normalize_goal(target_vec, target_keys)

    windows = list(iter_windows(args.start_annotation, chunk_size=args.chunk_size, stride=1))
    if not windows:
        raise SystemExit("no trajectory windows in start_annotation")
    start = windows[0].start
    img = _load_image_as_tensor(Path(start.image), tuple(args.resolution))

    batch = {"state_image": img.unsqueeze(0),
             "goal_vec": torch.from_numpy(target_norm).unsqueeze(0),
             "action_chunk": torch.zeros(1, args.chunk_size, 5)}
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
        obs_inputs, goal, _ = policy.prepare_inputs(batch, device, dtype)
        out = policy.sample(obs_inputs, goal, n_steps=args.n_steps)   # denormalized (m/rad)

    action_chunk = out.pred_action_chunk.squeeze(0).float().cpu().numpy()
    print(f"predicted {args.chunk_size}-step action chunk (m/rad):")
    for i, a in enumerate(action_chunk):
        print(f"  step {i}: dx={a[0]:.3f} dy={a[1]:.3f} dz={a[2]:.3f} dyaw={a[3]:.3f} dpitch={a[4]:.3f}")

    next_pos, next_fwd, next_up = apply_action_5d(
        np.asarray(start.camera_position, dtype=np.float32),
        np.asarray(start.camera_forward, dtype=np.float32),
        np.asarray(start.camera_up, dtype=np.float32),
        action_chunk[0],
    )
    print("starting pose:", start.camera_position)
    print("predicted next pose:", next_pos.tolist())

    proxy = None
    if {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}.issubset(target_keys):
        obj_pos = np.asarray(start.object_position, dtype=np.float32)
        vec = obj_pos - next_pos
        az = float(np.degrees(np.arctan2(vec[1], vec[0])))
        el = float(np.degrees(np.arctan2(vec[2], np.linalg.norm(vec[:2]))))
        pose_keys = [k for k in target_keys if k in {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}]
        achieved = np.array([az if k == "cam_to_obj_azimuth_deg" else el for k in pose_keys], dtype=np.float32)
        pose_target = np.array([target_dict[k] for k in pose_keys], dtype=np.float32)
        proxy = float(score_distance(achieved, pose_target, pose_keys))
        print(f"pose-proxy score distance to target (az/el only, normalized): {proxy:.4f}")

    json.dump(
        {"action_chunk": action_chunk.tolist(),
         "next_pose": {"position": next_pos.tolist(), "forward": next_fwd.tolist(), "up": next_up.tolist()},
         "proxy_distance": proxy, "n_steps": args.n_steps, "chunk_size": args.chunk_size},
        open(args.checkpoint.parent / f"eval_{args.start_annotation.stem}.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
