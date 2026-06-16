"""Evaluate the LLM Policy baseline on a target shot profile.

Mirrors `scripts/eval_vla_policy.py` / `eval_diffusion_policy.py` so all baselines
are directly comparable: same v7 start frame, same target yaml, same pose-proxy
distance metric and JSON output — only the policy differs (a VLM prompted for the
next camera move; training-free).

  python scripts/eval_llm_policy.py \
      --config configs/policy/llm_policy_qwen.yaml \
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
from src.policy.llm_policy.backends import build_backend
from src.policy.llm_policy.policy import LLMPhotoPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path, help="backend config (backend: qwen_local|api)")
    p.add_argument("--start_annotation", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path, help="YAML with `target:` {score_key: value}")
    p.add_argument("--chunk_size", type=int, default=8, help="only used to locate the start window")
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: next to start_annotation)")
    return p.parse_args()


def load_target_yaml(path: Path) -> dict[str, float]:
    cfg = yaml.safe_load(path.read_text())
    target = cfg.get("target", {})
    remap = {"center_x": "object_center_x", "center_y": "object_center_y"}
    return {remap.get(k, k): float(v) for k, v in target.items()}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    backend = build_backend(cfg)
    pol_cfg = cfg.get("policy", {})
    policy = LLMPhotoPolicy(
        backend,
        max_translation_m=float(pol_cfg.get("max_translation_m", 3.0)),
        max_angle_deg=float(pol_cfg.get("max_angle_deg", 60.0)),
    )

    target_dict = load_target_yaml(args.target)
    target_keys = goal_keys(list(target_dict.keys()))

    windows = list(iter_windows(args.start_annotation, chunk_size=args.chunk_size, stride=1))
    if not windows:
        raise SystemExit("no trajectory windows in start_annotation")
    start = windows[0].start
    image = Image.open(Path(start.image)).convert("RGB")

    action, info = policy.act(image, target_dict)
    print("LLM raw response:\n", info["raw"][:500])
    print(f"\nparsed action (m/rad): dx={action[0]:.3f} dy={action[1]:.3f} dz={action[2]:.3f} "
          f"dyaw={action[3]:.3f} dpitch={action[4]:.3f}")

    next_pos, next_fwd, next_up = apply_action_5d(
        np.asarray(start.camera_position, dtype=np.float32),
        np.asarray(start.camera_forward, dtype=np.float32),
        np.asarray(start.camera_up, dtype=np.float32),
        action,
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

    out_path = args.out or (args.start_annotation.parent / f"eval_llm_{args.start_annotation.stem}.json")
    json.dump(
        {"action": action.tolist(),
         "next_pose": {"position": next_pos.tolist(), "forward": next_fwd.tolist(), "up": next_up.tolist()},
         "proxy_distance": proxy, "llm_raw": info["raw"], "llm_parsed": info["parsed"]},
        open(out_path, "w"), indent=2,
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
