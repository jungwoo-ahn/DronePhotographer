"""Evaluate the UNIC baseline on a target shot profile.

Mirrors the other baseline evals (`eval_vla_policy.py`, `eval_diffusion_policy.py`,
`eval_llm_policy.py`): same v7 start frame, same target yaml, same pose-proxy
distance metric and JSON output. UNIC is goal-agnostic (it recommends a generically
well-composed crop, ignoring the target), so a high pose-proxy distance to a
*specified* target is the expected, intended contrast.

  python scripts/eval_unic_policy.py \
      --checkpoint weights/unic/unic_pretrained.pth \
      --start_annotation data/trajectories_full/<placement>/data.json \
      --target configs/policy/targets/centered_medium.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.policy.common.action_repr import apply_action_5d
from src.policy.common.annotations import iter_windows
from src.policy.common.goal_space import goal_keys
from src.policy.common.reward import score_distance
from src.policy.unic.model import UNICModel
from src.policy.unic.policy import UNICPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path, help="UNIC pretrained .pth")
    p.add_argument("--start_annotation", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--hfov_deg", type=float, default=50.0, help="render camera horizontal FOV")
    p.add_argument("--chunk_size", type=int, default=8, help="only used to locate the start window")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_target_yaml(path: Path) -> dict[str, float]:
    cfg = yaml.safe_load(path.read_text())
    target = cfg.get("target", {})
    remap = {"center_x": "object_center_x", "center_y": "object_center_y"}
    return {remap.get(k, k): float(v) for k, v in target.items()}


def main() -> None:
    args = parse_args()
    model = UNICModel.load(args.checkpoint, device=args.device, use_ema=args.use_ema)
    policy = UNICPolicy(model, hfov_deg=args.hfov_deg)

    target_dict = load_target_yaml(args.target)
    target_keys = goal_keys(list(target_dict.keys()))

    windows = list(iter_windows(args.start_annotation, chunk_size=args.chunk_size, stride=1))
    if not windows:
        raise SystemExit("no trajectory windows in start_annotation")
    start = windows[0].start
    image = Image.open(Path(start.image)).convert("RGB")

    action, info = policy.act(image)
    rec = info["recommendation"]
    print(f"UNIC box (normalized): center=({rec['center_x']:.3f},{rec['center_y']:.3f}) "
          f"size=({rec['width']:.3f},{rec['height']:.3f}) score={rec['score']:.3f}")
    print(f"action (m/rad): dx={action[0]:.3f} dy={action[1]:.3f} dz={action[2]:.3f} "
          f"dyaw={action[3]:.3f} dpitch={action[4]:.3f}")

    next_pos, next_fwd, next_up = apply_action_5d(
        np.asarray(start.camera_position, dtype=np.float32),
        np.asarray(start.camera_forward, dtype=np.float32),
        np.asarray(start.camera_up, dtype=np.float32),
        action,
    )

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

    out_path = args.out or (args.start_annotation.parent / f"eval_unic_{args.start_annotation.stem}.json")
    json.dump(
        {"action": action.tolist(),
         "next_pose": {"position": next_pos.tolist(), "forward": next_fwd.tolist(), "up": next_up.tolist()},
         "proxy_distance": proxy, "unic_recommendation": rec},
        open(out_path, "w"), indent=2,
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
