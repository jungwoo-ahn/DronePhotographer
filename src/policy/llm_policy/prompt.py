"""Prompt construction for the LLM Policy baseline (Photo Agent style).

The target shot profile (the same score-key dict the other baselines consume) is
rendered into a natural-language framing description — language is the LLM's
native goal interface — and the model is asked for the next camera move as a
strict JSON action in our camera-local 5D convention.
"""

from __future__ import annotations

from typing import Mapping

from src.policy.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH

SYSTEM_PROMPT = (
    "You are an expert photographer controlling a camera drone. You see the camera's "
    "current view and a description of the desired framing of the subject. You reason "
    "about how moving the camera would change the framing (you must imagine the result "
    "before moving), then output a single next camera move.\n\n"
    "The camera's local axes (from the camera's point of view): +x = right, +y = up, "
    "+z = forward (toward the scene). Translations are in metres. Rotations: yaw turns "
    "the camera left/right (positive = pan right), pitch tilts it up/down (positive = "
    "tilt up); both in degrees. There is no roll.\n\n"
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"reasoning": "<one short sentence>", "dx": <float>, "dy": <float>, '
    '"dz": <float>, "dyaw_deg": <float>, "dpitch_deg": <float>}'
)


def _phrase(key: str, value: float) -> str | None:
    """Map one (score_key, value) to a human framing phrase. Returns None to skip."""
    v = float(value)
    if key == "occupancy":
        size = ("a wide/establishing shot" if v < 15 else "a medium shot" if v < 45
                else "a close-up" if v < 75 else "an extreme close-up")
        return f"the subject should fill about {v:.0f}% of the frame ({size})"
    if key == "body_in_frame_ratio":
        return ("the entire subject should be inside the frame" if v >= 95
                else f"about {v:.0f}% of the subject should be in frame")
    if key == "cam_to_obj_azimuth_deg":
        return f"view the subject from an azimuth of about {v:.0f}° around it"
    if key == "cam_to_obj_elevation_deg":
        ang = ("eye level" if abs(v) < 7 else f"a high angle (~{abs(v):.0f}° above)" if v < 0
               else f"a low angle (~{v:.0f}° below)")
        return f"shoot from {ang}"
    if key == "object_center_x":
        frac = v / RENDER_WIDTH
        where = ("the left third" if frac < 0.4 else "horizontally centered" if frac < 0.6 else "the right third")
        return f"place the subject in {where} of the frame"
    if key == "object_center_y":
        frac = v / RENDER_HEIGHT
        where = ("the upper third" if frac < 0.4 else "vertically centered" if frac < 0.6 else "the lower third")
        return f"place the subject in {where} of the frame"
    if key in ("bbox_x_offset", "bbox_y_offset"):
        return None  # size cues already conveyed by occupancy; skip to avoid noise
    return f"{key} = {v:g}"


def describe_goal(target: Mapping[str, float]) -> str:
    """Render a target score-profile dict into a natural-language framing brief."""
    parts = [p for k, v in target.items() if (p := _phrase(k, v)) is not None]
    if not parts:
        return "Frame the subject well."
    return "Desired framing: " + "; ".join(parts) + "."


def build_user_prompt(target: Mapping[str, float]) -> str:
    """The per-step user message: the framing brief + the action request."""
    return (
        f"{describe_goal(target)}\n\n"
        "Given the current view above, output the single best next camera move to get "
        "closer to this framing, as the JSON object specified. Keep moves modest "
        "(translations within a couple of metres, rotations within ~45°)."
    )


__all__ = ["SYSTEM_PROMPT", "describe_goal", "build_user_prompt"]
