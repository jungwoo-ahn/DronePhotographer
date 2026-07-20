"""5D camera-action representation: (Δx, Δy, Δz, Δyaw, Δpitch).

All quantities are in the previous frame's camera-local basis (Blender convention:
+X=right, +Y=up, -Z=forward). Roll is dropped — drones don't roll the camera in
this project (see legacy `disable_roll: true` MPC configs).

Encoding direction conventions:
- Translation components are (right, up, forward) in metres.
- Yaw is rotation around local +Y (up); positive = nose right.
- Pitch is rotation around local +X (right); positive = nose up.
- Both angles are in radians, applied as R = R_yaw(α) @ R_pitch(β).
"""

from __future__ import annotations

import numpy as np

from src.utils.rotation_utils import (
    relative_rotation_matrix_camera_local,
    relative_translation_camera_local,
    rotvec_to_rotation_matrix,
)

ACTION_DIM = 5

# Per-dimension scale used to normalize the 5D action into ~[-1, 1] before it is
# tile-injected into the Cosmos action latent frame. Each entry is the p99 of |Δ|
# over the TRAINING sampling scheme's per-step actions: [right, up, forward (m),
# yaw, pitch (rad)].
#
# Fit for `sampling_scheme=multiscale_bidir` (offsets 8/16/24), whose strided
# offset-16/24 actions merge 2-3 real steps — so the rotation p99s (yaw/pitch) are
# ~2.3-2.8x the old single-step values ([0.32,0.28,0.95,0.081,0.071]); using that
# single-step scale here would clip most rotation for far goals. metres-vs-radians
# differ ~10x, so normalization is essential for the flow-matching L2 to weight all
# dims fairly. Measured over an 80-placement sample (~576k per-step actions) with
# `scripts/fit_action_scale.py --sampling multiscale_bidir`; recompute on the full
# data (via sbatch) to refine. One-line swap here (or pass `action_scale=`).
ACTION_SCALE: np.ndarray = np.array(
    [0.552, 0.316, 0.969, 0.223, 0.164], dtype=np.float32
)


def normalize_action_5d(action: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Map a raw 5D action (metres / radians) to ~[-1, 1] by per-dim scaling + clip."""
    s = ACTION_SCALE if scale is None else np.asarray(scale, dtype=np.float32)
    return np.clip(np.asarray(action, dtype=np.float32) / s, -1.0, 1.0).astype(np.float32)


def denormalize_action_5d(action: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Inverse of `normalize_action_5d` (clipping is not inverted)."""
    s = ACTION_SCALE if scale is None else np.asarray(scale, dtype=np.float32)
    return (np.asarray(action, dtype=np.float32) * s).astype(np.float32)


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
    """Encode the pose delta as a 5D action in the previous frame's camera-local basis."""
    dt = relative_translation_camera_local(prev_position, next_position, prev_forward, prev_up)
    rel = relative_rotation_matrix_camera_local(prev_forward, prev_up, next_forward, next_up)
    yaw, pitch = yaw_pitch_from_rotation_matrix(rel)
    return np.array([dt[0], dt[1], dt[2], yaw, pitch], dtype=np.float32)


def decode_action_5d(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a 5D action into (Δtranslation_camera_local_xyz, Δrotation_matrix_camera_local)."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}-D action, got shape {a.shape}")
    dt = a[:3].astype(np.float32)
    rot = rotation_matrix_from_yaw_pitch(float(a[3]), float(a[4]))
    return dt, rot


def apply_action_5d(
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a 5D action to a pose; returns (next_position, next_forward, next_up)."""
    from src.utils.rotation_utils import apply_camera_local_action, rotation_matrix_to_rotvec

    dt, rot = decode_action_5d(action)
    rotvec = rotation_matrix_to_rotvec(rot)
    return apply_camera_local_action(position, forward, up, dt, rotvec)


__all__ = [
    "ACTION_DIM",
    "ACTION_SCALE",
    "normalize_action_5d",
    "denormalize_action_5d",
    "yaw_pitch_from_rotation_matrix",
    "rotation_matrix_from_yaw_pitch",
    "encode_action_5d",
    "decode_action_5d",
    "apply_action_5d",
]
