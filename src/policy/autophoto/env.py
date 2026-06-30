"""PhotoEnv — AutoPhoto's RL environment, reimplemented over our Blender rollout env.

A `gymnasium.Env` that mirrors AutoPhoto's setup (9 discrete nav actions, aesthetic
features as the observation, a score-gradient reward with time/exploration shaping,
a terminate action) but drives our `BlenderRolloutEnv` + `AestheticReward` instead
of Habitat. The policy is trained from scratch in here (SB3); the reward model is
the reused pretrained scorer. Goal-agnostic by construction (no target profile).

Action map (AutoPhoto `fine_turns`) -> our camera-local 5D action (metres / radians):
  0 FORWARD  +0.25 dz     3 TERMINATE (stop)     6 SMALL_RIGHT +10° yaw
  1 TURN_LEFT  -30° yaw   4 MOVE_BACK -0.25 dz    7 LARGE_LEFT  -90° yaw
  2 TURN_RIGHT +30° yaw   5 SMALL_LEFT -10° yaw   8 LARGE_RIGHT +90° yaw
(yaw+ = nose right, so "left" turns are negative.)

Terminal reward: we use the dense improvement `final_score - init_score` (AutoPhoto's
distance-based ±1 needs a per-scene sample DB we don't carry; noted in REFERENCES).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # SB3 stack optional until training
    raise ImportError("PhotoEnv needs gymnasium (pip install stable-baselines3 sb3-contrib)") from e

_FWD = 0.25
_T10, _T30, _T90 = math.radians(10), math.radians(30), math.radians(90)


@dataclass(frozen=True)
class _Action:
    name: str
    delta: tuple                # 5D (dx, dy, dz, dyaw, dpitch) raw
    terminate: bool = False


PHOTO_ACTIONS: list[_Action] = [
    _Action("forward", (0, 0, _FWD, 0, 0)),
    _Action("turn_left", (0, 0, 0, -_T30, 0)),
    _Action("turn_right", (0, 0, 0, _T30, 0)),
    _Action("terminate", (0, 0, 0, 0, 0), terminate=True),
    _Action("move_back", (0, 0, -_FWD, 0, 0)),
    _Action("small_left", (0, 0, 0, -_T10, 0)),
    _Action("small_right", (0, 0, 0, _T10, 0)),
    _Action("large_left", (0, 0, 0, -_T90, 0)),
    _Action("large_right", (0, 0, 0, _T90, 0)),
]


class PhotoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rollout_env, reward, *, pair_idx: int = 0, max_steps: int = 100,
                 gamma: float = 0.9999, time_penalty: float = 0.005,
                 exploration_weight: float = 0.1) -> None:
        super().__init__()
        self.rollout = rollout_env
        self.reward = reward
        self.pair_idx = pair_idx
        self.max_steps = max_steps
        self.gamma = gamma
        self.time_penalty = time_penalty
        self.exploration_weight = exploration_weight

        self.action_space = spaces.Discrete(len(PHOTO_ACTIONS))
        dim = int(getattr(reward, "feature_dim", 512))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

        self.t = 0
        self.global_time = 0
        self.init_score = 0.0
        self.prev_score = 0.0
        self.current_score = 0.0
        self._last_obs = np.zeros(dim, dtype=np.float32)

    def _observe(self, image) -> tuple[np.ndarray, float]:
        score, feat = self.reward.score_and_features(image)
        self._last_obs = np.asarray(feat, dtype=np.float32)
        return self._last_obs, float(score)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.rollout.reset_to_start(self.pair_idx, render=True)
        feat, score = self._observe(obs_dict["image"])
        self.t = 0
        self.init_score = self.prev_score = self.current_score = score
        return feat, {"score": score, "init_score": score}

    def step(self, action: int):
        spec = PHOTO_ACTIONS[int(action)]
        if spec.terminate:
            reward = self.current_score - self.init_score   # dense improvement terminal
            info = {"final_score": self.current_score, "init_score": self.init_score,
                    "better_than_init": self.current_score > self.init_score}
            return self._last_obs, float(reward), True, False, info

        obs_dict, _ = self.rollout.step(np.asarray(spec.delta, dtype=np.float32), render=True)
        feat, score = self._observe(obs_dict["image"])
        score_grad = score - self.prev_score
        exploration = self.exploration_weight * (self.gamma ** self.global_time)
        reward = score_grad + exploration - self.time_penalty * self.t
        self.prev_score = self.current_score = score
        self.t += 1
        self.global_time += 1
        truncated = self.t >= self.max_steps
        info = {"score": score, "score_grad": score_grad}
        return feat, float(reward), False, bool(truncated), info

    def close(self):
        self.rollout.close()


__all__ = ["PhotoEnv", "PHOTO_ACTIONS"]
