"""UNICPolicy — the UNIC composition model used as a reactive camera policy.

UNIC recommends a composition box for the current view. We turn that recommendation
into one of our camera-local 5D actions:

  - **pan** (recommended box center offset from frame center) -> camera rotation:
    a box centered right/below means rotate to recenter it (yaw right / pitch down).
    Offsets are converted to angles via the camera field of view.
  - **zoom** (recommended box width vs the full frame) -> dolly `dz`: a tighter
    recommended crop (width < 1) means move forward; a box spilling past the
    borders (width > 1) means back off.

Faithful goal-agnostic: UNIC has no notion of our target shot profile — it
recommends a generically well-composed crop and reacts only to the current frame
(no previsualization of unseen viewpoints). Lateral translation is left at zero
(pan is realized as rotation; depth is unknown so dolly is a monotonic heuristic).
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from src.policy.common.action_repr import ACTION_DIM
from src.policy.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH


class _Recommender(Protocol):
    def recommend(self, image): ...


class UNICPolicy:
    """Wrap a UNIC recommender as a reactive 5D camera policy.

    Args:
      model: anything with `recommend(image) -> UNICRecommendation` (the real
        `UNICModel`, or a mock in tests).
      hfov_deg: horizontal field of view of the render camera (sets how a box-center
        offset maps to a rotation angle). vFOV is derived from the frame aspect.
      dolly_gain: metres of dolly per unit of (1 - recommended_width).
      max_translation_m / max_angle_deg: defensive per-step clamps.
    """

    def __init__(self, model: _Recommender, *, hfov_deg: float = 50.0,
                 aspect: float = RENDER_WIDTH / RENDER_HEIGHT, dolly_gain: float = 1.5,
                 max_translation_m: float = 3.0, max_angle_deg: float = 60.0) -> None:
        self.model = model
        self.hfov_deg = hfov_deg
        self.aspect = aspect
        self.dolly_gain = dolly_gain
        self.max_translation_m = max_translation_m
        self.max_angle_deg = max_angle_deg

    def _vfov_deg(self) -> float:
        h = math.tan(math.radians(self.hfov_deg) / 2.0) / self.aspect
        return math.degrees(2.0 * math.atan(h))

    def act(self, image) -> tuple[np.ndarray, dict]:
        rec = self.model.recommend(image)
        ang = math.radians(self.max_angle_deg)
        t = self.max_translation_m

        # pan -> rotation. Box center right of frame center (dx>0) -> yaw right (+);
        # below center (dy>0, image y is down) -> pitch down (-).
        dx = rec.center_x - 0.5
        dy = rec.center_y - 0.5
        yaw = math.radians(dx * self.hfov_deg)
        pitch = -math.radians(dy * self._vfov_deg())
        # zoom -> dolly forward when the recommended crop is tighter than the frame.
        dz = self.dolly_gain * (1.0 - rec.width)

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[2] = float(np.clip(dz, -t, t))
        action[3] = float(np.clip(yaw, -ang, ang))
        action[4] = float(np.clip(pitch, -ang, ang))
        return action, {"recommendation": vars(rec)}


__all__ = ["UNICPolicy"]
