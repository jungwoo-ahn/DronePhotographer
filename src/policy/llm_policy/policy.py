"""LLMPhotoPolicy — the LLM Policy baseline (Photo Agent style).

A training-free policy: prompt a VLM with the current frame + a natural-language
framing brief (rendered from the target shot profile), parse a single next camera
move, and return it as our camera-local 5D action `(dx, dy, dz, dyaw, dpitch)` in
metres / radians — the same action space the trainable baselines use, so the eval
and pose-distance metric are shared.

This is the "LLM previsualizes implicitly" baseline: the model is asked to imagine
the effect of a move before committing, with no explicit world model and no
training. Backend is pluggable (`Qwen3-VL-2B` placeholder, OpenAI-compatible API
for final runs) — see `backends.py`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.policy.common.action_repr import ACTION_DIM
from src.policy.llm_policy.backends import VLMBackend
from src.policy.llm_policy.prompt import SYSTEM_PROMPT, build_user_prompt
from src.vlm.api import parse_vlm_json

_FIELDS = ("dx", "dy", "dz", "dyaw_deg", "dpitch_deg")


class LLMPhotoPolicy:
    """VLM-as-policy. `act(image, target) -> (action_5d_raw, info)`.

    Args:
      backend: a `VLMBackend` (local Qwen or API).
      max_translation_m / max_angle_deg: per-step clamps applied to the parsed
        move (defensive — the model is also asked to keep moves modest).
    """

    def __init__(self, backend: VLMBackend, *, max_translation_m: float = 3.0, max_angle_deg: float = 60.0) -> None:
        self.backend = backend
        self.max_translation_m = max_translation_m
        self.max_angle_deg = max_angle_deg

    def _parse_action(self, parsed: dict | None) -> np.ndarray:
        """JSON dict -> 5D action (m / rad), clamped. Missing/garbage -> zeros."""
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        if not parsed:
            return a
        vals = []
        for f in _FIELDS:
            try:
                vals.append(float(parsed.get(f, 0.0)))
            except (TypeError, ValueError):
                vals.append(0.0)
        dx, dy, dz, dyaw_deg, dpitch_deg = vals
        t = self.max_translation_m
        ang = np.deg2rad(self.max_angle_deg)
        a[0] = float(np.clip(dx, -t, t))
        a[1] = float(np.clip(dy, -t, t))
        a[2] = float(np.clip(dz, -t, t))
        a[3] = float(np.clip(np.deg2rad(dyaw_deg), -ang, ang))
        a[4] = float(np.clip(np.deg2rad(dpitch_deg), -ang, ang))
        return a

    def act(self, image, target: Mapping[str, float]) -> tuple[np.ndarray, dict]:
        """Predict the next camera move from the current `image` (PIL) + target profile."""
        user = build_user_prompt(target)
        text = self.backend.generate([image], SYSTEM_PROMPT, user)
        parsed = parse_vlm_json(text)
        action = self._parse_action(parsed)
        return action, {"raw": text, "parsed": parsed}


__all__ = ["LLMPhotoPolicy"]
