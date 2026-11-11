"""The shots the operator can pick in the panel, and what each asks for in words.

A shot profile is the policy's goal: where the subject should sit in the frame, how much of
it, and from which side. The positions are in the renders' 1024x768 pixels, which is also the
view the laptop sends, so a profile can be drawn straight over "what the model sees".

The list comes from the evaluation episodes, eight from each side of the subject, because
those are goals the policy was measured on rather than ones made up here.
"""

from __future__ import annotations

import json
from pathlib import Path

FRAME_W, FRAME_H = 1024.0, 768.0
PROFILE_KEYS = ("occupancy", "body_in_frame_ratio", "subject_bearing_deg",
                "cam_to_obj_elevation_deg", "object_center_x", "object_center_y",
                "bbox_x_offset", "bbox_y_offset")
CROP_KEYS = ("top_cut_frac", "bot_cut_frac", "visible_frac")
EPISODES = Path(__file__).resolve().parents[2] / "configs" / "eval_episodes_v1.json"
SIDES = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")


def sentence(profile: dict) -> str:
    """The exact sentence the server turns this profile into for the model."""
    import numpy as np

    from src.policy.common.goal_text import goal_prompt

    crop = {k: float(profile[k]) for k in CROP_KEYS if k in profile} or None
    return goal_prompt(np.array([float(profile[k]) for k in PROFILE_KEYS]), crop=crop)


def short_name(profile: dict) -> str:
    """A few words to tell the shots apart in a list."""
    from src.policy.common.facing import sector8
    from src.policy.common.goal_text import PLACE_X, SHOT_SIZE, _classify

    size = _classify(float(profile["occupancy"]), SHOT_SIZE)
    place = _classify(float(profile["object_center_x"]) / FRAME_W, PLACE_X)
    side = sector8(float(profile["subject_bearing_deg"]))
    return f"{size[0].upper()}{size[1:]} from the {side}, {place}, {float(profile['occupancy']):.0f}%"


def _shot(shot_id: str, group: str, profile: dict, name: str | None = None) -> dict:
    return {"id": shot_id, "group": group, "name": name or short_name(profile),
            "profile": profile, "sentence": sentence(profile)}


def complete(profile: dict) -> bool:
    return all(k in profile for k in PROFILE_KEYS)


def episode_shots(path: Path = EPISODES) -> list[dict]:
    """One shot per evaluation episode, grouped by the side of the subject it is taken from."""
    with open(path, encoding="utf-8") as f:
        episodes = json.load(f)["episodes"]
    shots = []
    for i, e in enumerate(episodes):
        raw = e["goal_raw"]
        profile = {k: float(raw[k]) for k in PROFILE_KEYS if k in raw}
        profile["subject_bearing_deg"] = float(e["bearing"])
        profile.update({k: float(raw[k]) for k in CROP_KEYS if k in raw})
        if complete(profile):
            shots.append(_shot(f"eval-{i:02d}", f"From the {e['sector']}", profile))
    order = {f"From the {s}": n for n, s in enumerate(SIDES)}
    shots.sort(key=lambda s: (order.get(s["group"], 99), s["profile"]["occupancy"]))
    return shots


def load_shots(goal: dict | None = None, episodes: Path | None = EPISODES) -> list[dict]:
    """The goal the pilot was started with first, then the evaluation episodes' goals."""
    shots = []
    if goal and complete(goal.get("profile", {})):
        p = goal["profile"]
        shots.append(_shot("start", "Started with", p, "The goal file: " + short_name(p)))
    if episodes is not None and Path(episodes).exists():
        shots.extend(episode_shots(Path(episodes)))
    return shots


__all__ = ["load_shots", "episode_shots", "sentence", "short_name", "complete",
           "FRAME_W", "FRAME_H", "PROFILE_KEYS"]
