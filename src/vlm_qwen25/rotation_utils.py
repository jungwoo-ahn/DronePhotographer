from __future__ import annotations

import numpy as np


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + eps)


def make_camera_rotation_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build camera-to-world rotation matrix (3x3).

    Convention matches Blender camera axes:
    - local +X: right
    - local +Y: up
    - local -Z: forward (view direction)
    """
    fwd = normalize(forward)
    upn = normalize(up)
    right = normalize(np.cross(fwd, upn))
    upn = normalize(np.cross(right, fwd))
    return np.stack([right, upn, -fwd], axis=1)


def relative_rotation_matrix(
    forward_i: np.ndarray,
    up_i: np.ndarray,
    forward_j: np.ndarray,
    up_j: np.ndarray,
) -> np.ndarray:
    r_i = make_camera_rotation_from_forward_up(forward_i, up_i)
    r_j = make_camera_rotation_from_forward_up(forward_j, up_j)
    return r_j @ r_i.T


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (SO(3)) to axis-angle vector (radians)."""
    r = rotation.astype(np.float64)
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)

    w, x, y, z = quat.tolist()
    vec = np.array([x, y, z], dtype=np.float64)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-10:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * float(np.arctan2(vec_norm, w))
    axis = vec / vec_norm

    # Keep angle in [0, pi] for a stable canonical rotvec.
    if angle > np.pi:
        angle = 2.0 * np.pi - angle
        axis = -axis

    return (axis * angle).astype(np.float32)


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector (radians) to rotation matrix (SO(3))."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = rotvec / theta
    x, y, z = axis.tolist()
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    identity = np.eye(3, dtype=np.float32)
    return identity + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def relative_rotation_rotvec(
    forward_i: np.ndarray,
    up_i: np.ndarray,
    forward_j: np.ndarray,
    up_j: np.ndarray,
) -> np.ndarray:
    rel = relative_rotation_matrix(forward_i, up_i, forward_j, up_j)
    return rotation_matrix_to_rotvec(rel).astype(np.float32)


def rotation_quality(rotation: np.ndarray) -> tuple[float, float]:
    """Return (det_error, orthogonality_error)."""
    det_error = abs(float(np.linalg.det(rotation)) - 1.0)
    orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    return det_error, orth_error
