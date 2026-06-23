"""Train the AutoPhoto baseline (issue #22): RL with an aesthetic reward, in Blender.

Reuses AutoPhoto's pretrained aesthetic scorer as the reward (src/policy/autophoto/
reward.py) and trains a fresh RecurrentPPO policy (matching AutoPhoto's PPO+LSTM)
in our Blender rollout env (PhotoEnv). Goal-agnostic. Reduced scale + EEVEE
persistent rendering make it tractable (see REFERENCES.md).

  python scripts/train_autophoto.py --config configs/policy/autophoto.yaml

Runs on the render machine (needs the blender binary + scenes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--total_timesteps", type=int, default=None)
    return p.parse_args()


def _load_placements(manifest: str | list) -> list[str]:
    if isinstance(manifest, list):
        return manifest
    return [ln.strip() for ln in Path(manifest).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    total = args.total_timesteps or int(cfg["train"]["total_timesteps"])

    from sb3_contrib import RecurrentPPO

    from src.policy.autophoto.env import PhotoEnv
    from src.policy.autophoto.renderer import PersistentBlenderRenderer
    from src.policy.autophoto.reward import AestheticReward
    from src.policy.common.blender_env import BlenderRolloutEnv
    from src.policy.common.validation_sample import load_validation_sample

    reward = AestheticReward(cfg["reward"]["checkpoint"], device=cfg["reward"].get("device", "cuda"))
    placements = _load_placements(cfg["data"]["placements"])
    vlm_dir = cfg["data"]["vlm_placements_dir"]
    rcfg = cfg.get("renderer", {})

    class SceneCyclingPhotoEnv(PhotoEnv):
        """PhotoEnv that swaps to the next placement every `scene_change_freq` episodes."""

        def __init__(self):
            self._placements = placements
            self._freq = int(cfg["train"].get("scene_change_freq", 5))
            self._ep = 0
            self._idx = 0
            rollout = self._build_rollout(0)
            super().__init__(rollout, reward, max_steps=int(cfg["train"].get("max_steps", 100)))

        def _build_rollout(self, idx):
            sample = load_validation_sample(self._placements[idx], vlm_dir)
            renderer = PersistentBlenderRenderer(
                engine=rcfg.get("engine", "BLENDER_EEVEE_NEXT"), samples=int(rcfg.get("samples", 16)))
            return BlenderRolloutEnv.from_validation_sample(sample, renderer)

        def reset(self, *, seed=None, options=None):
            if self._ep and self._ep % self._freq == 0 and len(self._placements) > 1:
                self.rollout.close()
                self._idx = (self._idx + 1) % len(self._placements)
                self.rollout = self._build_rollout(self._idx)
            self._ep += 1
            return super().reset(seed=seed, options=options)

    env = SceneCyclingPhotoEnv()
    out_dir = Path(cfg["train"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    model = RecurrentPPO(
        "MlpLstmPolicy", env, verbose=1,
        n_steps=int(cfg["train"].get("n_steps", 256)),
        learning_rate=float(cfg["train"].get("learning_rate", 3e-4)),
        tensorboard_log=str(out_dir / "tb"),
    )
    # Periodic checkpoints: Blender-in-the-loop RL is slow (~8-11s/env-step), so a
    # multi-hour run must survive interruption. Save every `save_freq` env steps.
    from stable_baselines3.common.callbacks import CheckpointCallback

    save_freq = int(cfg["train"].get("save_freq", 2000))
    cb = CheckpointCallback(save_freq=save_freq, save_path=str(out_dir / "ckpts"),
                            name_prefix="autophoto")
    model.learn(total_timesteps=total, callback=cb)
    model.save(str(out_dir / "autophoto_policy"))
    env.close()
    print("saved policy ->", out_dir / "autophoto_policy.zip")


if __name__ == "__main__":
    main()
