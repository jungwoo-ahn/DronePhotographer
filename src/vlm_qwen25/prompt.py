from __future__ import annotations

from typing import Sequence


def action_vector_to_text(
    delta_position: tuple[float, float, float],
    delta_rotation: tuple[float, float, float],
    action_frame: str = "camera_local",
) -> str:
    dx, dy, dz = delta_position
    rx, ry, rz = delta_rotation

    if action_frame == "camera_local":
        return (
            f"move_camera_local_m(right={dx:.4f}, up={dy:.4f}, forward={dz:.4f}); "
            f"rotate_camera_local_axis_angle_rad(rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f})"
        )
    if action_frame == "world":
        return (
            f"move_world_m(x={dx:.4f}, y={dy:.4f}, z={dz:.4f}); "
            f"rotate_world_axis_angle_rad(rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f})"
        )
    raise ValueError(f"unsupported action_frame: {action_frame}")


def no_action_text(action_frame: str = "camera_local") -> str:
    return action_vector_to_text(
        delta_position=(0.0, 0.0, 0.0),
        delta_rotation=(0.0, 0.0, 0.0),
        action_frame=action_frame,
    )


def build_user_prompt(
    action_text: str,
    target_score_keys: Sequence[str],
    action_frame: str = "camera_local",
) -> str:
    keys_text = ", ".join(target_score_keys)
    if action_frame == "camera_local":
        frame_text = "camera-local frame of image_i (+right, +up, +forward)."
    elif action_frame == "world":
        frame_text = "world frame."
    else:
        raise ValueError(f"unsupported action_frame: {action_frame}")

    return (
        "You are a drone image scorer.\n"
        f"Input is current image_i and an action command in {frame_text}\n"
        "Predict score values of image_j after applying the action.\n"
        f"Action: {action_text}\n"
        f"Return only a JSON object with keys exactly in this order: {keys_text}\n"
        "Use numeric values only. No extra text."
    )
