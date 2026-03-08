from __future__ import annotations


def action_vector_to_text(delta_position: tuple[float, float, float], delta_rotation: tuple[float, float, float]) -> str:
    dx, dy, dz = delta_position
    rx, ry, rz = delta_rotation
    return (
        f"move(x={dx:.4f}, y={dy:.4f}, z={dz:.4f}); "
        f"rotate_axis_angle(rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f})"
    )


def build_user_prompt(action_text: str) -> str:
    return (
        "You are a drone composition scorer.\n"
        "Input is current image_i and an action command.\n"
        "Predict the composition scores of image_j after applying the action.\n"
        f"Action: {action_text}\n"
        "Return only a JSON object with keys exactly in this order:\n"
        '{"rule_of_thirds_line":float,"breathing_space":float,'
        '"centeredness":float,"subject_size_20":float,"subject_size_80":float}\n'
        "Each value must be in [0, 1]. No extra text."
    )
