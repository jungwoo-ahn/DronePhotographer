"""Evaluate the trained AutoPhoto policy on a validation sample.

Goal-agnostic: rolls the policy out in the Blender env and reports the aesthetic
score it achieves (its own objective) + improvement over the start, plus the same
pose-proxy distance to a target the other baselines report (expected to be poor,
since AutoPhoto optimizes aesthetics, not a specified shot profile).

  python scripts/eval_autophoto.py --config configs/policy/autophoto.yaml \
      --policy runs/autophoto/autophoto_policy.zip \
      --start_annotation data/trajectories_full/<placement>/data.json \
      --target configs/policy/targets/centered_medium.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--start_annotation", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from sb3_contrib import RecurrentPPO

    from src.policy.autophoto.env import PhotoEnv
    from src.policy.autophoto.renderer import PersistentBlenderRenderer
    from src.policy.autophoto.reward import AestheticReward
    from src.policy.common.blender_env import BlenderRolloutEnv
    from src.policy.common.goal_space import goal_keys
    from src.policy.common.validation_sample import load_validation_sample

    reward = AestheticReward(cfg["reward"]["checkpoint"], device=cfg["reward"].get("device", "cuda"))
    sample = load_validation_sample(args.start_annotation, cfg["data"]["vlm_placements_dir"])
    rcfg = cfg.get("renderer", {})
    renderer = PersistentBlenderRenderer(engine=rcfg.get("engine", "BLENDER_EEVEE_NEXT"),
                                         samples=int(rcfg.get("samples", 16)))
    rollout = BlenderRolloutEnv.from_validation_sample(sample, renderer)
    env = PhotoEnv(rollout, reward, max_steps=args.max_steps)
    model = RecurrentPPO.load(str(args.policy))

    obs, info = env.reset()
    init_score = info["init_score"]
    lstm_state, done = None, False
    while not done:
        action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    final_pos = env.rollout.position

    target = {(("object_center_x" if k == "center_x" else "object_center_y" if k == "center_y" else k)): float(v)
              for k, v in yaml.safe_load(args.target.read_text()).get("target", {}).items()}
    keys = goal_keys(list(target))
    proxy = None
    if {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}.issubset(keys):
        from src.policy.common.blender_env import pose_proxy_distance
        proxy = pose_proxy_distance(final_pos, np.asarray(sample.subject_center), target, keys)

    final_score = env.current_score
    print(f"init_score={init_score:.4f}  final_score={final_score:.4f}  "
          f"improvement={final_score - init_score:+.4f}  pose_proxy={proxy}")
    out = args.out or (args.start_annotation.parent / f"eval_autophoto_{args.start_annotation.stem}.json")
    json.dump({"init_score": init_score, "final_score": final_score,
               "improvement": final_score - init_score, "proxy_distance": proxy,
               "final_pose": final_pos.tolist()}, open(out, "w"), indent=2)
    print("wrote", out)
    env.close()


if __name__ == "__main__":
    main()
