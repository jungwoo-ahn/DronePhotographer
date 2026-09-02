"""5D camera-action representation: (Δright, Δup, Δforward, Δyaw, Δpitch).

Translation (Δright, Δup, Δforward) is in the previous frame's camera-local basis
(Blender convention: +X=right, +Y=up, -Z=forward), in metres.

Rotation is (Δyaw, Δpitch) of where the camera AIMS, in radians:
- Δyaw   = change in the forward vector's WORLD azimuth — a rotation about the
           world up axis (+Z), positive = turn left→right about vertical.
- Δpitch = change in the forward vector's elevation — a rotation about the camera's
           local right axis (horizontal for a level camera), positive = aim up.

Why yaw about WORLD up, not camera-local up: the trajectories are level (bank == 0
relative to world up — verified in scripts/verify_conventions.py), so a camera's
orientation has only two real DOF, (azimuth, elevation). Yawing about the camera's
*local* up axis while the camera is pitched induces roll (the up vector banks),
which this 5D representation cannot encode — so applying such an action would miss
the true (level) endpoint, badly for large moves (offset-24 chunks were ~11°/0.5m
off). Yawing about the *world* up axis keeps the camera level, so every real
endpoint is reachable with zero roll and an encoded action chunk composes back onto
the endpoint exactly.
"""

from __future__ import annotations

import numpy as np

from src.utils.rotation_utils import (
    camera_rotation_opencv_from_forward_up,
    forward_up_from_camera_rotation_opencv,
    matrix_from_rot6d,
    orthonormalize_forward_up,
    project_forward_up_upright,
    relative_translation_camera_local,
    rot6d_from_matrix,
    rotvec_to_rotation_matrix,
    translation_camera_local_to_world,
)

# The pose action is jungwoo's V12 9D: [Δtranslation_cam(3), rot6d(6)] — left RAW
# (unnormalized): translation std <=0.14 m and rot6d entries in [-1,1] already sit at a
# commensurate scale (~3.4x), so ACTION_SCALE is all ones (normalize == identity). The
# full ACTION the policies predict is 10D: the 9 pose dims + a `shoot` channel (dim 9),
# a latched 0/1 "arrived, take the photo" state (see `dataset_base.shoot_column`).
POSE_DIM = 9
SHOOT_DIM = 1
ACTION_DIM = POSE_DIM + SHOOT_DIM  # 10
POSE_DIM_5D = 5                    # legacy 5D (Δright,Δup,Δfwd,Δyaw,Δpitch); kept for eval/reference

# World up axis (Blender +Z). Yaw rotates about this so a level camera stays level.
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# World up axis (Blender +Z). Yaw rotates about this so a level camera stays level.
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _forward_azimuth_elevation(forward: np.ndarray) -> tuple[float, float]:
    """(azimuth about world +Z, elevation above the horizon) of a forward vector, radians."""
    f = np.asarray(forward, dtype=np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-12:
        return 0.0, 0.0
    f = f / n
    return float(np.arctan2(f[1], f[0])), float(np.arcsin(max(-1.0, min(1.0, f[2]))))

# Per-dimension scale used to normalize the 5D action into ~[-1, 1] before it is
# tile-injected into the Cosmos action latent frame. Each entry is the p99 of |Δ|
# over the TRAINING sampling scheme's per-step actions: [right, up, forward (m),
# yaw, pitch (rad)].
#
# Fit for `sampling_scheme=multiscale_bidir` (offsets 8/16/24), whose strided
# offset-16/24 actions merge 2-3 real steps, so the p99s are 2-3x single-step values;
# using a single-step scale would clip most rotation for far goals. metres-vs-radians
# differ ~10x, so normalization is essential for the flow-matching L2 to weight all
# dims fairly. The yaw entry (0.223 -> 0.295) was refit after the world-up-yaw action
# change (Δyaw is now the forward's world-azimuth delta, whose p99 differs from the
# old camera-local-yaw); translation is unchanged by that change. Measured over a
# 120-file multiscale_bidir sample (~917k per-step actions) with
# `scripts/fit_action_scale.py`; recompute on the full data (via sbatch) to refine.
# 9D-raw scheme (jungwoo V12): the action is left UNNORMALIZED, so ACTION_SCALE is all
# ones and normalize/denormalize are the identity on the pose dims. Kept as a (POSE_DIM,)
# vector so the shoot-aware normalize_action_5d split still works.
ACTION_SCALE: np.ndarray = np.ones(POSE_DIM, dtype=np.float32)

# Per-dim STANDARD DEVIATION of the raw action over the multiscale_bidir scheme — the
# alternative "unit-variance" normalization. The Cosmos policy divides by this (instead of
# the p99 ACTION_SCALE) and clips at ±4 sigma, so each of the 5 dims contributes equally to
# the flow-matching L2 and the action signal is ~unit-scale (commensurate with the VAE image
# latents / flow noise). Diffusion / VLA keep the p99 ACTION_SCALE + [-1,1] clip. Refit with
# `scripts/fit_action_scale.py --stat std`.
ACTION_STD: np.ndarray = np.ones(POSE_DIM, dtype=np.float32)


def normalize_action_5d(
    action: np.ndarray, scale: np.ndarray | None = None, clip: float = 1.0
) -> np.ndarray:
    """Map a raw 5D action (metres / radians) to a bounded latent by per-dim scaling + clip.

    Default (`scale=ACTION_SCALE` p99, `clip=1.0`) maps to [-1, 1] — the diffusion / VLA
    convention. The Cosmos policy passes `scale=ACTION_STD, clip=4.0` for unit-variance
    normalization: each dim becomes ~N(0, 1), clipped at ±4 sigma so the ~0.006% tail past
    4 sigma is bounded without cutting real signal.
    """
    s = ACTION_SCALE if scale is None else np.asarray(scale, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32)
    # 6D action (pose + shoot): scale/clip the POSE_DIM pose dims, pass the 0/1 shoot
    # channel through untouched (it is already bounded and must stay {0, 1}).
    if a.shape[-1] == ACTION_DIM and s.shape[-1] == POSE_DIM:
        pose = np.clip(a[..., :POSE_DIM] / s, -clip, clip)
        return np.concatenate([pose, a[..., POSE_DIM:]], axis=-1).astype(np.float32)
    return np.clip(a / s, -clip, clip).astype(np.float32)


def denormalize_action_5d(action: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Inverse of `normalize_action_5d` (clipping is not inverted)."""
    s = ACTION_SCALE if scale is None else np.asarray(scale, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32)
    if a.shape[-1] == ACTION_DIM and s.shape[-1] == POSE_DIM:
        return np.concatenate([a[..., :POSE_DIM] * s, a[..., POSE_DIM:]], axis=-1).astype(np.float32)
    return (a * s).astype(np.float32)


def yaw_pitch_from_rotation_matrix(rot: np.ndarray) -> tuple[float, float]:
    """Decompose a camera-local rotation R = R_yaw(α) @ R_pitch(β) into (α, β).

    Roll is assumed zero — any non-zero roll in the input silently projects away.
    """
    r = np.asarray(rot, dtype=np.float64)
    # R[1,2] = -sin β, R[1,1] = cos β
    pitch = float(np.arctan2(-r[1, 2], r[1, 1]))
    # R[2,0] = -sin α cos β, R[0,0] = cos α cos β
    yaw = float(np.arctan2(-r[2, 0], r[0, 0]))
    return yaw, pitch


def rotation_matrix_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """Build R = R_yaw(α) @ R_pitch(β) — rotation about local +Y then local +X."""
    ca, sa = np.cos(yaw), np.sin(yaw)
    cb, sb = np.cos(pitch), np.sin(pitch)
    r_yaw = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]], dtype=np.float32)
    return (r_yaw @ r_pitch).astype(np.float32)


def encode_action_5d(
    prev_position: np.ndarray,
    prev_forward: np.ndarray,
    prev_up: np.ndarray,
    next_position: np.ndarray,
    next_forward: np.ndarray,
    next_up: np.ndarray,
) -> np.ndarray:
    """Encode the pose delta as a 5D action.

    Translation is the camera-local (right, up, forward) displacement; rotation is
    the change in the forward vector's world azimuth (Δyaw) and elevation (Δpitch).
    `apply_action_5d` inverts this exactly for level cameras (bank == 0).
    """
    dt = relative_translation_camera_local(prev_position, next_position, prev_forward, prev_up)
    az0, el0 = _forward_azimuth_elevation(prev_forward)
    az1, el1 = _forward_azimuth_elevation(next_forward)
    dyaw = float((az1 - az0 + np.pi) % (2.0 * np.pi) - np.pi)   # shortest signed turn
    dpitch = el1 - el0
    return np.array([dt[0], dt[1], dt[2], dyaw, dpitch], dtype=np.float32)


def decode_action_5d(action: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Split a 5D action into (Δtranslation_camera_local, (Δyaw, Δpitch)).

    Unlike a pure local rotation, Δyaw is about the WORLD up axis, so the rotation
    is pose-dependent and cannot be reduced to a standalone matrix — applying it
    needs the current pose (`apply_action_5d`). This helper validates + splits.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] not in (POSE_DIM, ACTION_DIM):
        raise ValueError(f"expected {POSE_DIM}- or {ACTION_DIM}-D action, got shape {a.shape}")
    return a[:3].astype(np.float32), (float(a[3]), float(a[4]))


def apply_action_5d(
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a 5D action to a pose; returns (next_position, next_forward, next_up).

    Translation moves along the current camera-local basis. Rotation pitches the
    forward about the (horizontal) local right axis by Δpitch, then yaws about the
    WORLD up axis by Δyaw — so a level camera stays level (no induced roll) and a
    zero rotation preserves the orientation exactly, including at the poles.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] not in (POSE_DIM, ACTION_DIM):
        raise ValueError(f"expected {POSE_DIM}- or {ACTION_DIM}-D action, got shape {a.shape}")
    # a[5] (shoot) is a policy output, not motion — ignored here.
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    dt = a[:3].astype(np.float64)
    dyaw, dpitch = float(a[3]), float(a[4])

    # translation in the current camera-local basis (unchanged from the legacy repr)
    delta_world = translation_camera_local_to_world(dt, forward, up)
    next_position = np.asarray(position, dtype=np.float64) + delta_world

    # rotation: pitch about the local right axis (horizontal for a level camera),
    # then yaw about the WORLD up axis. r_pitch @ forward first, then r_yaw.
    right = np.cross(forward, WORLD_UP)
    if np.linalg.norm(right) < 1e-6:                 # looking near-vertical: fall back to
        right = np.cross(forward, up)                # the camera's own right axis
    right = right / (np.linalg.norm(right) + 1e-12)
    r_pitch = rotvec_to_rotation_matrix((right * dpitch).astype(np.float32))
    r_yaw = rotvec_to_rotation_matrix((WORLD_UP * dyaw).astype(np.float32))
    rot = r_yaw @ r_pitch
    next_forward, next_up = orthonormalize_forward_up(
        (rot @ forward).astype(np.float32), (rot @ up).astype(np.float32)
    )
    return next_position.astype(np.float32), next_forward, next_up


# ---------------------------------------------------------------------------- #
# 9D action (V12: [Δtranslation_cam(3), rot6d(6)]) — the active representation.
# ---------------------------------------------------------------------------- #
def encode_action_9d(prev_position, prev_forward, prev_up,
                     next_position, next_forward, next_up) -> np.ndarray:
    """Encode a pose pair as the 9D action [Δtranslation(3), rot6d(6)] (V12 port).

    `apply_action_9d` inverts this exactly. Translation is in the OpenCV camera frame
    of the previous pose; rotation is the first two columns of the relative rotation.
    """
    rot_prev = camera_rotation_opencv_from_forward_up(prev_forward, prev_up)
    rot_next = camera_rotation_opencv_from_forward_up(next_forward, next_up)
    delta_rot = rot_prev.T @ rot_next
    delta_world = np.asarray(next_position, dtype=np.float64) - np.asarray(prev_position, dtype=np.float64)
    delta_translation = rot_prev.T @ delta_world
    return np.concatenate([delta_translation, rot6d_from_matrix(delta_rot)]).astype(np.float32)


def decode_action_9d(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 9D (or 10D pose+shoot) action into (Δtranslation_cam, relative 3x3 rotation)."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] not in (POSE_DIM, ACTION_DIM):
        raise ValueError(f"expected {POSE_DIM}- or {ACTION_DIM}-D action, got shape {a.shape}")
    return a[:3].astype(np.float32), matrix_from_rot6d(a[3:9])


def apply_action_9d(position, forward, up, action, upright: bool = True):
    """Apply a 9D action to a pose; returns (next_position, next_forward, next_up).

    `upright=True` re-projects the decoded pose roll-free (aim kept, bank removed).
    a[9] (shoot) is a policy output, not motion — ignored here.
    """
    delta_translation, delta_rot = decode_action_9d(action)
    rot_cur = camera_rotation_opencv_from_forward_up(forward, up)
    next_position = np.asarray(position, dtype=np.float64) + rot_cur @ delta_translation.astype(np.float64)
    next_forward, next_up = forward_up_from_camera_rotation_opencv(rot_cur @ delta_rot)
    if upright:
        next_forward, next_up = project_forward_up_upright(next_forward, next_up)
    else:
        next_forward, next_up = orthonormalize_forward_up(next_forward, next_up)
    return next_position.astype(np.float32), next_forward, next_up


__all__ = [
    "ACTION_DIM",
    "POSE_DIM",
    "SHOOT_DIM",
    "ACTION_SCALE",
    "normalize_action_5d",
    "denormalize_action_5d",
    "yaw_pitch_from_rotation_matrix",
    "rotation_matrix_from_yaw_pitch",
    "encode_action_5d",
    "decode_action_5d",
    "apply_action_5d",
    "encode_action_9d",
    "decode_action_9d",
    "apply_action_9d",
]
