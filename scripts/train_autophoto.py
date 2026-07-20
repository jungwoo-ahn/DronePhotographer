"""Train the AutoPhoto baseline (issue #22): RL with an aesthetic reward, in Blender.

Reuses AutoPhoto's pretrained aesthetic scorer as the reward (src/policy/autophoto/
reward.py) and trains a fresh RecurrentPPO policy (matching AutoPhoto's PPO+LSTM)
in our Blender rollout env (PhotoEnv). Goal-agnostic. Reduced scale + EEVEE
persistent rendering make it tractable (see REFERENCES.md).

  python scripts/train_autophoto.py --config configs/policy/autophoto.yaml

Rollouts are render-bound (~4-11s/env-step), so training parallelizes across
`train.n_envs` Blender workers (SubprocVecEnv, one persistent Blender + reward
model per env, spread over `train.env_gpus`). Runs on the render machine.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--resume_from", type=str, default=None,
                   help="SB3 .zip checkpoint to resume from (continues num_timesteps)")
    return p.parse_args()


def _load_placements(manifest: str | list) -> list[str]:
    """Resolve the train placements: an inline list, a .yaml/.json manifest
    ({placements: [...]} or a bare list), or a legacy .txt (one path per line)."""
    if isinstance(manifest, (list, tuple)):
        return list(manifest)
    path = Path(manifest)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        doc = yaml.safe_load(path.read_text())
        return doc["placements"] if isinstance(doc, dict) and "placements" in doc else list(doc)
    if suffix == ".json":
        import json
        doc = json.loads(path.read_text())
        return doc["placements"] if isinstance(doc, dict) and "placements" in doc else list(doc)
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _make_scene_cycling_env(placements, vlm_dir, rcfg, reward, freq, max_steps):
    """Build a PhotoEnv that swaps to the next placement every `freq` episodes."""
    from src.policy.autophoto.env import PhotoEnv
    from src.policy.autophoto.renderer import PersistentBlenderRenderer
    from src.policy.common.blender_env import BlenderRolloutEnv
    from src.policy.common.validation_sample import load_validation_sample

    class SceneCyclingPhotoEnv(PhotoEnv):
        def __init__(self):
            self._placements = placements
            self._freq = freq
            self._ep = 0
            self._idx = 0
            self._force_advance = False
            rollout = self._build_rollout_skipping_bad(0)
            super().__init__(rollout, reward, max_steps=max_steps)

        def _build_rollout(self, idx):
            sample = load_validation_sample(self._placements[idx], vlm_dir)
            renderer = PersistentBlenderRenderer(
                engine=rcfg.get("engine", "BLENDER_EEVEE_NEXT"), samples=int(rcfg.get("samples", 16)),
                cycles_gpu=bool(rcfg.get("cycles_gpu", False)),
                timeout_s=float(rcfg.get("timeout_s", 90.0)))
            return BlenderRolloutEnv.from_validation_sample(sample, renderer)

        def _build_rollout_skipping_bad(self, idx):
            """A broken placement (e.g. unreadable .blend) must not kill a multi-day
            run: try successive placements, skipping any whose scene fails to load.
            The first render is forced here so load errors surface NOW, not later."""
            for k in range(len(self._placements)):
                j = (idx + k) % len(self._placements)
                try:
                    rollout = self._build_rollout(j)
                    rollout.reset_to_start(0, render=True)
                    self._idx = j
                    return rollout
                except Exception as e:  # noqa: BLE001
                    print(f"[autophoto] skipping bad placement {self._placements[j]}: "
                          f"{str(e)[:200]}", flush=True)
            raise RuntimeError("no loadable placement in this env's shard")

        def reset(self, *, seed=None, options=None):
            if (self._force_advance or (self._ep and self._ep % self._freq == 0)) and len(self._placements) > 1:
                try:
                    self.rollout.close()
                except Exception:  # noqa: BLE001
                    pass
                self.rollout = self._build_rollout_skipping_bad((self._idx + 1) % len(self._placements))
            self._force_advance = False
            self._ep += 1
            # A reset render can still fail (e.g. render timeout); advance until one works.
            for _ in range(len(self._placements)):
                try:
                    return super().reset(seed=seed, options=options)
                except Exception as e:  # noqa: BLE001
                    print(f"[autophoto] reset failed on {self._placements[self._idx]}: "
                          f"{str(e)[:150]}; advancing scene", flush=True)
                    try:
                        self.rollout.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self.rollout = self._build_rollout_skipping_bad((self._idx + 1) % len(self._placements))
            raise RuntimeError("no placement produced a valid reset in this shard")

        def step(self, action):
            # A render failure mid-episode (e.g. hung EEVEE -> RenderTimeout) must NOT
            # propagate: an uncaught exception kills this SubprocVecEnv worker, which
            # then deadlocks the whole trainer. End the episode + advance scene instead.
            try:
                return super().step(action)
            except Exception as e:  # noqa: BLE001
                print(f"[autophoto] step render failed on {self._placements[self._idx]}: "
                      f"{str(e)[:150]}; ending episode", flush=True)
                self._force_advance = True
                return self._last_obs, 0.0, True, False, {"render_error": True}

    return SceneCyclingPhotoEnv()


class EnvFactory:
    """Picklable per-worker env builder for SubprocVecEnv (spawn).

    Holds only plain data; everything heavy (reward model, Blender worker) is
    constructed inside the child process. `gpu_id` pins BOTH the reward model and
    the child's Blender/EEVEE to one physical GPU via CUDA_VISIBLE_DEVICES, set
    before the first CUDA touch in that process.
    """

    def __init__(self, placements, vlm_dir, rcfg, reward_ckpt, freq, max_steps, gpu_id=None):
        self.placements = list(placements)
        self.vlm_dir = vlm_dir
        self.rcfg = dict(rcfg)
        self.reward_ckpt = reward_ckpt
        self.freq = freq
        self.max_steps = max_steps
        self.gpu_id = gpu_id

    def __call__(self):
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        from src.policy.autophoto.reward import AestheticReward

        reward = AestheticReward(self.reward_ckpt, device="cuda")
        return _make_scene_cycling_env(
            self.placements, self.vlm_dir, self.rcfg, reward, self.freq, self.max_steps)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    total = args.total_timesteps or int(cfg["train"]["total_timesteps"])

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    placements = _load_placements(cfg["data"]["placements"])
    vlm_dir = cfg["data"]["vlm_placements_dir"]
    rcfg = cfg.get("renderer", {})
    tcfg = cfg["train"]
    freq = int(tcfg.get("scene_change_freq", 5))
    max_steps = int(tcfg.get("max_steps", 100))
    n_envs = int(tcfg.get("n_envs", 1))
    gpus = tcfg.get("env_gpus") or [None]

    if n_envs > 1:
        # Disjoint placement shards per env: parallel envs see different scenes
        # (more diverse rollouts) and never contend for the same Blender worker.
        factories = [
            EnvFactory(placements[i::n_envs], vlm_dir, rcfg, cfg["reward"]["checkpoint"],
                       freq, max_steps, gpu_id=gpus[i % len(gpus)])
            for i in range(n_envs)
        ]
        env = SubprocVecEnv(factories, start_method="spawn")
    else:
        from src.policy.autophoto.reward import AestheticReward

        reward = AestheticReward(cfg["reward"]["checkpoint"], device=cfg["reward"].get("device", "cuda"))
        env = _make_scene_cycling_env(placements, vlm_dir, rcfg, reward, freq, max_steps)

    out_dir = Path(tcfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        model = RecurrentPPO.load(args.resume_from, env=env,
                                  tensorboard_log=str(out_dir / "tb"))
        print(f"resumed from {args.resume_from} at num_timesteps={model.num_timesteps}", flush=True)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1,
            n_steps=int(tcfg.get("n_steps", 256)),
            learning_rate=float(tcfg.get("learning_rate", 3e-4)),
            tensorboard_log=str(out_dir / "tb"),
        )
    # Periodic checkpoints: multi-day run must survive interruption. save_freq is
    # counted in vec-env steps (n_envs timesteps each), so divide to keep the
    # cadence ~`save_freq` TOTAL env steps.
    save_freq = max(1, int(tcfg.get("save_freq", 2000)) // max(1, n_envs))
    cb = CheckpointCallback(save_freq=save_freq, save_path=str(out_dir / "ckpts"),
                            name_prefix="autophoto")
    # reset_num_timesteps=False on resume: `total` stays the CUMULATIVE budget.
    model.learn(total_timesteps=total, callback=cb,
                reset_num_timesteps=not args.resume_from)
    model.save(str(out_dir / "autophoto_policy"))
    env.close()
    print("saved policy ->", out_dir / "autophoto_policy.zip")


if __name__ == "__main__":
    main()
