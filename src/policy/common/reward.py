"""Score-distance reward / value target.

Per issue #18, the value head's target is the **score distance** between the
achieved profile and the goal profile. We default to a weighted L2 (lower is
better) and expose it as a *reward* (negative distance, so higher is better).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.policy.common.goal_space import goal_keys, normalize_goal


def score_distance(
    achieved: np.ndarray,
    goal: np.ndarray,
    keys: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Weighted L2 between achieved and goal profile vectors.

    By default both vectors are normalized to [-1, 1] per key before the L2,
    which keeps keys with different physical scales (degrees vs. percent)
    comparable. Pass `normalize=False` if the inputs are already on a shared
    scale.
    """
    keys = goal_keys(keys)
    a = np.asarray(achieved, dtype=np.float32)
    g = np.asarray(goal, dtype=np.float32)
    if normalize:
        a = normalize_goal(a, keys)
        g = normalize_goal(g, keys)
    diff = a - g
    if weights is None:
        return float(np.sqrt(np.mean(diff * diff)))
    w = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=np.float32)
    return float(np.sqrt(np.sum(w * diff * diff) / np.sum(w)))


def score_distance_reward(
    achieved: np.ndarray,
    goal: np.ndarray,
    keys: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Negative of `score_distance` — higher = better. Suitable as a value target."""
    return -score_distance(achieved, goal, keys, weights, normalize=normalize)
