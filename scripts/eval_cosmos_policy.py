"""Evaluate a trained Cosmos policy on a target shot profile.

Pipeline:
  1. Load checkpoint + Cosmos backbone via diffusers.
  2. Load a starting view from a v6 annotation.
  3. Encode the starting image with the VAE.
  4. Run the rectified-flow diffusion sampler conditioned on the goal vector.
  5. Extract the predicted action chunk + value from the action/value latent
     frames (cosmos-policy style: split into repeats, average across them).
  6. Apply the *first* predicted action to the camera pose to get the next pose.
  7. Optionally re-render in Blender to measure the realized goal-profile error.

Usage:
  python scripts/eval_cosmos_policy.py \\
      --checkpoint runs/<ts>_cosmos_2b_v6_proto/ckpt_last.pt \\
      --start_annotation outputs/.../andrew.json \\
      --target configs/inference/centered_medium.yaml \\
      [--n_steps 32] [--render]
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
from src.policy.cosmos.dataset import _load_image_as_tensor
from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.vae import CosmosVAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--start_annotation", required=True, type=Path,
                   help="v7 placement data.json; we use the first window's start frame as the starting state")
    p.add_argument("--target", required=True, type=Path,
                   help="YAML with `target:` mapping {score_key: value}")
    p.add_argument("--image_root", default=".", type=Path)
    p.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_steps", type=int, default=32, help="diffusion-sampler steps")
    p.add_argument("--chunk_size", type=int, default=1,
                   help="must match the chunk_size used at training")
    p.add_argument("--render", action="store_true",
                   help="invoke Blender to re-render at the predicted next pose")
    return p.parse_args()


def load_target_yaml(path: Path) -> dict[str, float]:
    cfg = yaml.safe_load(path.read_text())
    target = cfg.get("target", {})
    out: dict[str, float] = {}
    remap = {"center_x": "object_center_x", "center_y": "object_center_y"}
    for k, v in target.items():
        out[remap.get(k, k)] = float(v)
    return out


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2.5-2B", dtype=dtype, device_map=args.device,
    )
    vae = CosmosVAEWrapper(pipe.vae).to(device).eval()
    target_dict = load_target_yaml(args.target)
    policy = CosmosWorldActionPolicy(
        pipe.transformer,
        goal_dim=len(target_dict),
        chunk_size=args.chunk_size,
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
    # v7 image paths are resolved to absolute by iter_windows
    img_path = Path(start.image) if Path(start.image).is_absolute() else args.image_root / start.image
    image = _load_image_as_tensor(img_path, tuple(args.resolution)).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        # Build the conditioning clip the same way the trainer does: state repeated
        # in both halves of the 4-frame VAE chunk (no "next" view at inference time).
        clip = vae.assemble_clip(image, image)
        image_latent = vae.encode(clip)
        goal_tensor = torch.from_numpy(target_norm).unsqueeze(0).to(device, dtype=dtype)
        out = policy.sample(image_latent=image_latent, goal_vec=goal_tensor, n_steps=args.n_steps)

    # sample() already returns the action chunk in physical units (it denormalizes
    # by the model's action_scale buffer loaded from the checkpoint).
    action_chunk = out.pred_action_chunk.squeeze(0).float().cpu().numpy()   # (chunk_size, 5)
    pred_value = float(out.pred_value.squeeze(0)) if out.pred_value is not None else None

    print(f"predicted {args.chunk_size}-step action chunk (m/rad):")
    for i, a in enumerate(action_chunk):
        print(f"  step {i}: dx={a[0]:.3f} dy={a[1]:.3f} dz={a[2]:.3f} dyaw={a[3]:.3f} dpitch={a[4]:.3f}")
    print(f"predicted value (~ -score_distance to goal): {pred_value}")

    # Apply only the first action of the chunk to get the next pose
    next_pos, next_fwd, next_up = apply_action_5d(
        np.asarray(start.camera_position, dtype=np.float32),
        np.asarray(start.camera_forward, dtype=np.float32),
        np.asarray(start.camera_up, dtype=np.float32),
        action_chunk[0],
    )
    print("starting pose:", start.camera_position)
    print("predicted next pose:", next_pos.tolist())

    if args.render:
        raise NotImplementedError(
            "Blender rollout TODO: subprocess.run with src/drones/blender_drone.py + apply_action_5d"
        )

    proxy = None
    if {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}.issubset(target_keys):
        obj_pos = np.asarray(start.object_position, dtype=np.float32)
        vec = obj_pos - next_pos
        az = float(np.degrees(np.arctan2(vec[1], vec[0])))
        el = float(np.degrees(np.arctan2(vec[2], np.linalg.norm(vec[:2]))))
        pose_keys = [k for k in target_keys if k in {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}]
        achieved = np.array(
            [az if k == "cam_to_obj_azimuth_deg" else el for k in pose_keys],
            dtype=np.float32,
        )
        pose_target = np.array([target_dict[k] for k in pose_keys], dtype=np.float32)
        proxy = float(score_distance(achieved, pose_target, pose_keys))
        print(f"pose-proxy score distance to target (az/el only, normalized): {proxy:.4f}")

    json.dump(
        {
            "action_chunk": action_chunk.tolist(),
            "value": pred_value,
            "next_pose": {"position": next_pos.tolist(), "forward": next_fwd.tolist(), "up": next_up.tolist()},
            "proxy_distance": proxy,
            "n_steps": args.n_steps,
            "chunk_size": args.chunk_size,
        },
        open(args.checkpoint.parent / f"eval_{args.start_annotation.stem}.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
