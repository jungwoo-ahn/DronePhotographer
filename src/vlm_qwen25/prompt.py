from __future__ import annotations

from typing import Sequence


def action_vector_to_text(delta_position: tuple[float, float, float], delta_rotation: tuple[float, float, float]) -> str:
    dx, dy, dz = delta_position
    rx, ry, rz = delta_rotation
    return (
        f"move(x={dx:.4f}, y={dy:.4f}, z={dz:.4f}); "
        f"rotate_axis_angle(rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f})"
    )


def build_user_prompt(action_text: str, target_score_keys: Sequence[str]) -> str:
    keys_text = ", ".join(target_score_keys)
    return (
        "You are a drone image scorer.\n"
        "Input is current image_i and an action command.\n"
        "Predict score values of image_j after applying the action.\n"
        f"Action: {action_text}\n"
        f"Return only a JSON object with keys exactly in this order: {keys_text}\n"
        "Use numeric values only. No extra text."
    )
