"""The PhotoFlow Reviewer: rule checks, visual critique, scalar J, pairwise incumbent choice.

Paper (arXiv:2605.23771):

  m1 in {0,1}  subject is inside the frame AND in the coarse screen half the composition
               rule asks for.
  m2           max(0, 1 - d / 0.45), d = euclidean screen-space distance from the projected
               subject centre to the requested composition point.
  m3..m6       VLM scores in [0,1]: composition, technical quality, aesthetic appeal,
               alignment with the instruction and the soft blueprint.

  J = 0.10*m1 + 0.10*m2 + 0.15*m3 + 0.15*m4 + 0.25*m5 + 0.25*m6

m1/m2 are computed from the *brief's* requested half and composition point -- the same
quantised vocabulary the brief is written in. They deliberately do NOT see the continuous
goal-distance the harness scores with: that would hand the search its own objective, since
our `d` is built from subject bearing, elevation and size.
"""

from __future__ import annotations

import numpy as np

M2_FALLOFF = 0.45
WEIGHTS = {"m1": 0.10, "m2": 0.10, "m3": 0.15, "m4": 0.15, "m5": 0.25, "m6": 0.25}
SEMANTIC_KEY = "m6"              # the "semantic score" the Reflector thresholds on


def rule_in_requested_half(center_xy, in_frame: bool, requested_half: str | None) -> float:
    """m1. `requested_half` is 'left'/'right'/'top'/'bottom'/None, from the brief."""
    if not in_frame:
        return 0.0
    u, v = float(center_xy[0]), float(center_xy[1])
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return 0.0
    if requested_half is None:
        return 1.0
    ok = {"left": u < 0.5, "right": u > 0.5, "top": v < 0.5, "bottom": v > 0.5}
    return 1.0 if ok.get(requested_half, True) else 0.0


def rule_composition_distance(center_xy, target_xy) -> float:
    """m2 = max(0, 1 - d/0.45) on the normalised screen-space distance."""
    d = float(np.linalg.norm(np.asarray(center_xy, float) - np.asarray(target_xy, float)))
    return max(0.0, 1.0 - d / M2_FALLOFF)


def combined_score(m: dict) -> float:
    """J(c). Missing dimensions count as 0 so a failed critique cannot inflate a candidate."""
    return float(sum(w * float(np.clip(m.get(k, 0.0), 0.0, 1.0)) for k, w in WEIGHTS.items()))


def pairwise_choose(incumbent, candidate):
    """Pairwise incumbent selection: 'better AND more stable', not greedy on J alone.

    The candidate replaces the incumbent only if it wins on J *and* does not lose any
    dimension badly -- a candidate that buys aesthetics by dropping the subject out of the
    requested half is exactly the trade the paper's pairwise step exists to refuse.
    Returns (winner, replaced).
    """
    if incumbent is None:
        return candidate, True
    if candidate is None:
        return incumbent, False
    if candidate["J"] <= incumbent["J"]:
        return incumbent, False
    ia, ca = incumbent.get("m", {}), candidate.get("m", {})
    if float(ca.get("m1", 0.0)) < float(ia.get("m1", 0.0)):
        return incumbent, False                       # lost the hard framing rule
    regressions = sum(1 for k in WEIGHTS
                      if float(ca.get(k, 0.0)) < float(ia.get(k, 0.0)) - 0.25)
    if regressions >= 2:
        return incumbent, False                       # won on average, unstable in detail
    return candidate, True


__all__ = ["rule_in_requested_half", "rule_composition_distance", "combined_score",
           "pairwise_choose", "WEIGHTS", "M2_FALLOFF", "SEMANTIC_KEY"]
