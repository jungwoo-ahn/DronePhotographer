from __future__ import annotations

from typing import Sequence


def action_text_from_rotvec(
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


def action_text_from_orientation_6d(
    delta_position: tuple[float, float, float],
    target_forward: tuple[float, float, float],
    target_up: tuple[float, float, float],
    action_frame: str = "camera_local",
) -> str:
    dx, dy, dz = delta_position
    fx, fy, fz = target_forward
    ux, uy, uz = target_up

    if action_frame == "camera_local":
        return (
            f"move_camera_local_m(right={dx:.4f}, up={dy:.4f}, forward={dz:.4f}); "
            f"orient_camera_local_6d(fx={fx:.4f}, fy={fy:.4f}, fz={fz:.4f}, "
            f"ux={ux:.4f}, uy={uy:.4f}, uz={uz:.4f})"
        )
    if action_frame == "world":
        return (
            f"move_world_m(x={dx:.4f}, y={dy:.4f}, z={dz:.4f}); "
            f"orient_world_6d(fx={fx:.4f}, fy={fy:.4f}, fz={fz:.4f}, "
            f"ux={ux:.4f}, uy={uy:.4f}, uz={uz:.4f})"
        )
    raise ValueError(f"unsupported action_frame: {action_frame}")


def build_action_text(
    *,
    delta_position: tuple[float, float, float],
    action_frame: str = "camera_local",
    rotation_representation: str = "orientation_6d",
    delta_rotation: tuple[float, float, float] | None = None,
    target_forward: tuple[float, float, float] | None = None,
    target_up: tuple[float, float, float] | None = None,
) -> str:
    rotation_representation = str(rotation_representation)
    if rotation_representation == "rotvec":
        if delta_rotation is None:
            raise ValueError("delta_rotation is required for rotation_representation='rotvec'")
        return action_text_from_rotvec(
            delta_position=delta_position,
            delta_rotation=delta_rotation,
            action_frame=action_frame,
        )
    if rotation_representation == "orientation_6d":
        if target_forward is None or target_up is None:
            raise ValueError("target_forward and target_up are required for rotation_representation='orientation_6d'")
        return action_text_from_orientation_6d(
            delta_position=delta_position,
            target_forward=target_forward,
            target_up=target_up,
            action_frame=action_frame,
        )
    raise ValueError(f"unsupported rotation_representation: {rotation_representation}")


def build_user_prompt(
    action_text: str,
    target_score_keys: Sequence[str],
    action_frame: str = "camera_local",
    rotation_representation: str = "orientation_6d",
) -> str:
    keys_text = ", ".join(target_score_keys)
    schema_text = "{" + ", ".join(f'"{key}": <float>' for key in target_score_keys) + "}"
    if action_frame == "camera_local":
        frame_text = "camera-local frame of image_i (+right, +up, +forward)."
    elif action_frame == "world":
        frame_text = "world frame."
    else:
        raise ValueError(f"unsupported action_frame: {action_frame}")

    if rotation_representation == "orientation_6d":
        orientation_text = "The action encodes the target camera orientation as target forward/up vectors."
    elif rotation_representation == "rotvec":
        orientation_text = "The action encodes rotation as an axis-angle rotation vector."
    else:
        raise ValueError(f"unsupported rotation_representation: {rotation_representation}")

    return (
        "You are a drone image scorer.\n"
        f"Input is current image_i and an action command in {frame_text}\n"
        f"{orientation_text}\n"
        "Predict score values of image_j after applying the action.\n"
        f"Action: {action_text}\n"
        "Return exactly one JSON object.\n"
        f"Use this JSON schema: {schema_text}\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        f"Use keys exactly once and in this exact order: {keys_text}\n"
        "Use numeric JSON values only. Do not use strings, null, comments, or trailing commas.\n"
        "Do not output markdown, code fences, explanations, or any extra text."
    )
