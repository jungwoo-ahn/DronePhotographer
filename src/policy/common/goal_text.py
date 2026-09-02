"""Goal -> cinematography-instruction text (V12 port of `lerobot_export.goal_prompt`).

Serializes an 8-key goal vector into the natural-language prompt jungwoo's V12 conditions
on: every key as a WORD and a NUMBER. The subject-bearing axis (`subject_bearing_deg`,
subject-relative) replaces the raw world azimuth, so "from the subject's back" is
meaningful. Used only by the VLA NL-conditioning path.

`NL_GOAL_KEYS` is the key order the returned vector/goal is expected in: our 8-key goal
space with `cam_to_obj_azimuth_deg` swapped for `subject_bearing_deg`.
"""
from __future__ import annotations

from typing import Collection, Mapping, Sequence

import numpy as np

from src.policy.common.facing import sector8
from src.policy.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH

SUBJECT_BEARING_KEY = "subject_bearing_deg"
NL_GOAL_KEYS = (
    "occupancy", "body_in_frame_ratio", SUBJECT_BEARING_KEY, "cam_to_obj_elevation_deg",
    "object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset",
)

# Cinematography vocabulary (raw units): {label: (lo, hi, centroid)}.
SHOT_SIZE = {
    "extreme wide shot": (0.0, 8.0, 4.0), "wide shot": (8.0, 20.0, 14.0),
    "medium-wide shot": (20.0, 38.0, 29.0), "medium shot": (38.0, 58.0, 48.0),
    "medium close-up": (58.0, 78.0, 68.0), "close-up": (78.0, 100.01, 88.0),
}
BODY_FRAMING = {
    "tightly cropped": (0.0, 30.0, 18.0), "partially cut off": (30.0, 60.0, 45.0),
    "mostly in frame": (60.0, 90.0, 75.0), "full body in frame": (90.0, 100.01, 96.0),
}
ELEVATION = {  # NEGATIVE = camera ABOVE the subject (looking down)
    "high angle": (-90.0, -20.0, -42.0), "eye level": (-20.0, 15.0, -3.0),
    "low angle": (15.0, 90.0, 33.0),
}
PLACE_X = {
    "off-screen left": (-9.0, 0.0, -0.12), "left third": (0.0, 0.38, 0.19),
    "centered": (0.38, 0.62, 0.5), "right third": (0.62, 1.0, 0.81),
    "off-screen right": (1.0, 9.0, 1.12),
}
PLACE_Y = {
    "off-screen top": (-9.0, 0.0, -0.12), "upper": (0.0, 0.38, 0.19),
    "mid": (0.38, 0.62, 0.5), "lower": (0.62, 1.0, 0.81),
    "off-screen bottom": (1.0, 9.0, 1.12),
}


def _classify(value: float, table: dict) -> str:
    for label, (lo, hi, _c) in table.items():
        if lo <= value < hi:
            return label
    return next(iter(table)) if value < 0 else list(table)[-1]  # clamp to an edge band


def crop_phrase(top_cut: float, bot_cut: float) -> str:
    """Which END of the subject the frame cuts (the one fact no goal key carries)."""
    top, bot = top_cut > 0.02, bot_cut > 0.02
    if top and bot:
        return "cropped at both the head and the feet"
    if top:
        return "cropped above the head"
    if bot:
        return "cropped below the waist" if bot_cut > 0.35 else "cropped at the legs"
    return "uncropped"


def goal_prompt(
    goal_vec: np.ndarray,
    keys: Sequence[str] = NL_GOAL_KEYS,
    *,
    crop: Mapping[str, float] | None = None,
    specified: Collection[str] | None = None,
) -> str:
    """The goal as a cinematography instruction: every key, as a word AND a number.

    Verbatim port of V12 `lerobot_export.goal_prompt`. The model receives the goal ONLY
    as this text. `keys` must place `subject_bearing_deg` in the bearing slot (NL_GOAL_KEYS).
    """
    v = {k: float(goal_vec[i]) for i, k in enumerate(keys)}
    bearing = v.get(SUBJECT_BEARING_KEY, 0.0)
    occ = v.get("occupancy", 0.0)
    body = v.get("body_in_frame_ratio", 0.0)
    elev = v.get("cam_to_obj_elevation_deg", 0.0)
    cx, cy = v.get("object_center_x", 0.0), v.get("object_center_y", 0.0)
    ox, oy = v.get("bbox_x_offset", 0.0), v.get("bbox_y_offset", 0.0)
    have = (lambda k: k in v) if specified is None else (lambda k: k in specified)

    head = (f"a {_classify(occ, SHOT_SIZE)} of the subject"
            if have("occupancy") else "a shot of the subject")
    if have(SUBJECT_BEARING_KEY):
        head += f" from the subject's {sector8(bearing)}"

    clauses, nums = [head], []
    if have("cam_to_obj_elevation_deg"):
        clauses.append(f"at {_classify(elev, ELEVATION)}")
    if have("object_center_x") and have("object_center_y"):
        clauses.append(f"{_classify(cx / RENDER_WIDTH, PLACE_X)} and "
                       f"{_classify(cy / RENDER_HEIGHT, PLACE_Y)} in the frame")
    elif have("object_center_x"):
        clauses.append(f"{_classify(cx / RENDER_WIDTH, PLACE_X)} in the frame")
    elif have("object_center_y"):
        clauses.append(f"{_classify(cy / RENDER_HEIGHT, PLACE_Y)} in the frame")
    if have("body_in_frame_ratio"):
        clauses.append(_classify(body, BODY_FRAMING))
    if crop is not None:
        clauses.append(crop_phrase(float(crop.get("top_cut_frac", 0.0)),
                                   float(crop.get("bot_cut_frac", 0.0))))

    if have(SUBJECT_BEARING_KEY):
        nums.append(f"bearing {bearing:.0f}°")
    if have("occupancy"):
        nums.append(f"occupancy {occ:.0f}%")
    if have("cam_to_obj_elevation_deg"):
        nums.append(f"elevation {elev:.0f}°")
    if have("body_in_frame_ratio"):
        nums.append(f"body_in_frame {body:.0f}%")
    if have("object_center_x") and have("object_center_y"):
        nums.append(f"center {cx:.0f}/{cy:.0f} px")
    if have("bbox_x_offset") and have("bbox_y_offset"):
        nums.append(f"half_size {ox:.0f}/{oy:.0f} px")
    if crop is not None and crop.get("visible_frac") is not None:
        nums.append(f"visible {float(crop['visible_frac']):.2f}")

    text = f"Move the camera to achieve this shot: {', '.join(clauses)}."
    return f"{text} ({', '.join(nums)})" if nums else text


__all__ = ["goal_prompt", "crop_phrase", "SUBJECT_BEARING_KEY", "NL_GOAL_KEYS"]
