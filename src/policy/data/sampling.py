"""Elliptical-orbit pair sampling for v7 trajectory dataset.

Pure numpy (no Blender). Implements:
- log-uniform random camera pose around a subject point O
- center-at-O ellipse closed-form through two endpoints
- constant true-anomaly parameterization
- a 32-frame look-at-O trajectory along the ellipse

See plan: /home/jungwooahn/.claude/plans/delegated-gathering-hoare.md
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

R_MIN: float = 0.8
R_MAX: float = 8.0
# Height-above-floor range. -0.1m allows camera fractionally below the floor
# (e.g. for subjects on a low rim); 2.5m is conservative vs. the project's
# is_camera_valid 3m floor-clearance check.
H_MIN_ABOVE_FLOOR: float = -0.1
H_MAX_ABOVE_FLOOR: float = 2.5
WORLD_UP: np.ndarray = np.array([0.0, 0.0, 1.0])

# Distance-dependent pitch jitter applied AFTER look-at. Format: (r_ref, lo_deg, hi_deg).
# Linear lerp between NEAR and FAR by radius; clipped outside.
# Close: camera can tilt up sharply (look up at subject). Far: small range.
# Matches v6 local-dense defaults.
PITCH_LERP_NEAR: tuple[float, float, float] = (1.0, -15.0, +45.0)
PITCH_LERP_FAR:  tuple[float, float, float] = (8.0, -15.0, +15.0)
# Distance-dependent yaw jitter (horizontal pan around WORLD_UP), applied BEFORE pitch.
# Close: wider left/right range (subject is large in frame). Far: narrow.
YAW_LERP_NEAR:   tuple[float, float, float] = (1.0, -30.0, +30.0)
YAW_LERP_FAR:    tuple[float, float, float] = (8.0, -10.0, +10.0)
K_CLIPS_PER_PLACEMENT: int = 12
MAX_ATTEMPTS: int = K_CLIPS_PER_PLACEMENT * 5
MIN_ANG_SEP_DEG: float = 15.0
MIDPOINT_TS: tuple[float, ...] = (0.25, 0.5, 0.75)
N_FRAMES: int = 32

_LOOKAT_SINGULARITY_EPS: float = 1e-6
_COLLINEAR_EPS: float = 1e-9
_DENOM_EPS: float = 1e-6


class DegenerateEllipse(ValueError):
    """Raised when (O, C_far, C_near) do not define a proper ellipse."""


class Pose(TypedDict):
    pos: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    r: float
    az: float
    elev: float
    pitch_jitter_deg: float
    yaw_jitter_deg: float


class Ellipse(TypedDict):
    O: np.ndarray
    u: np.ndarray
    v: np.ndarray
    a: float
    b: float
    theta_far: float
    theta_near: float


def _lerp_envelope(
    near: tuple[float, float, float],
    far: tuple[float, float, float],
    r: float,
) -> tuple[float, float]:
    r_n, lo_n, hi_n = near
    r_f, lo_f, hi_f = far
    span = r_f - r_n
    if span == 0.0:
        return float(lo_n), float(hi_n)
    t = (float(r) - r_n) / span
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    lo = lo_n + t * (lo_f - lo_n)
    hi = hi_n + t * (hi_f - hi_n)
    return float(lo), float(hi)


def pitch_range_at_r(r: float) -> tuple[float, float]:
    """Linearly interpolate (lo_deg, hi_deg) from PITCH_LERP_NEAR to PITCH_LERP_FAR.

    Clipped: r < r_near returns the NEAR range, r > r_far returns the FAR range.
    """
    return _lerp_envelope(PITCH_LERP_NEAR, PITCH_LERP_FAR, r)


def yaw_range_at_r(r: float) -> tuple[float, float]:
    """Linearly interpolate (lo_deg, hi_deg) from YAW_LERP_NEAR to YAW_LERP_FAR."""
    return _lerp_envelope(YAW_LERP_NEAR, YAW_LERP_FAR, r)


def apply_yaw(
    forward: np.ndarray, up: np.ndarray, yaw_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate forward/up around WORLD_UP by ``yaw_deg``.

    Positive yaw rotates the look-direction counterclockwise viewed from +z
    (subject shifts to the left of the frame). Inputs assumed unit & orthogonal.
    """
    if abs(yaw_deg) < 1e-9:
        return forward, up
    a = float(np.deg2rad(yaw_deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    k = WORLD_UP  # rotation axis
    # Rodrigues rotation around k=WORLD_UP.
    new_forward = (
        c * forward
        + s * np.cross(k, forward)
        + (1.0 - c) * float(np.dot(k, forward)) * k
    )
    new_forward = new_forward / float(np.linalg.norm(new_forward))
    new_up = (
        c * up
        + s * np.cross(k, up)
        + (1.0 - c) * float(np.dot(k, up)) * k
    )
    new_up = new_up / float(np.linalg.norm(new_up))
    return new_forward, new_up


def apply_pitch(
    forward: np.ndarray, up: np.ndarray, pitch_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate forward/up by ``pitch_deg`` around the camera's right axis.

    Positive pitch tilts the look-up component (camera looks higher); negative
    tilts down. Right axis = forward × WORLD_UP, then up is recomputed orthogonal.
    Returns (new_forward, new_up). Inputs are assumed unit and orthogonal.
    """
    if abs(pitch_deg) < 1e-9:
        return forward, up
    right = np.cross(forward, WORLD_UP)
    n = float(np.linalg.norm(right))
    if n < _LOOKAT_SINGULARITY_EPS:
        return forward, up
    right = right / n
    a = float(np.deg2rad(pitch_deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    # Rodrigues rotation of forward around axis=right by angle a.
    # forward' = c·forward + s·(right × forward) + (1−c)·(right · forward)·right
    # right is perpendicular to forward by construction, so dot = 0.
    new_forward = c * forward + s * np.cross(right, forward)
    new_forward = new_forward / float(np.linalg.norm(new_forward))
    # New up: keep right axis, recompute up from new_forward and right.
    new_up = np.cross(right, new_forward)
    new_up = new_up / float(np.linalg.norm(new_up))
    return new_forward, new_up


def _lookat(pos: np.ndarray, O: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """forward = (O - pos)/|·|; up = world_up minus parallel component to forward."""
    d = O - pos
    n = float(np.linalg.norm(d))
    if n < _LOOKAT_SINGULARITY_EPS:
        return None
    forward = d / n
    up_raw = WORLD_UP - WORLD_UP.dot(forward) * forward
    nu = float(np.linalg.norm(up_raw))
    if nu < _LOOKAT_SINGULARITY_EPS:
        return None
    return forward, up_raw / nu


def sample_pose(
    O: np.ndarray,
    rng: np.random.Generator,
    floor_z: Optional[float] = None,
) -> Optional[Pose]:
    """Sample one (pos, forward, up): log-uniform 3D radius, photographer-height z.

    Parameterization: az uniform, r 3D-radius log-uniform in [R_MIN, R_MAX], and
    camera z chosen so cam_z − floor_z ∈ [H_MIN_ABOVE_FLOOR, H_MAX_ABOVE_FLOOR].
    When `floor_z` is None, the constraint is applied relative to O.z instead
    (i.e. the floor is assumed to pass through O). `elev` is reported as
    arcsin((cam_z − O.z) / r) for downstream logging.

    Returns None if the radius/floor constraints are jointly infeasible, or if
    the resulting look-at is singular (forward ∥ WORLD_UP).
    """
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    r = float(np.exp(rng.uniform(np.log(R_MIN), np.log(R_MAX))))
    ref_z = float(O[2]) if floor_z is None else float(floor_z)
    dz_lo = max(ref_z + H_MIN_ABOVE_FLOOR - float(O[2]), -r)
    dz_hi = min(ref_z + H_MAX_ABOVE_FLOOR - float(O[2]),  r)
    if dz_hi <= dz_lo:
        return None
    dz = float(rng.uniform(dz_lo, dz_hi))
    r_xy_sq = r * r - dz * dz
    if r_xy_sq <= 0.0:
        return None
    r_xy = float(np.sqrt(r_xy_sq))
    cam = O + np.array([r_xy * np.cos(az), r_xy * np.sin(az), dz])
    look = _lookat(cam, O)
    if look is None:
        return None
    forward, up = look
    elev = float(np.arcsin(np.clip(dz / r, -1.0, 1.0)))
    pitch_lo, pitch_hi = pitch_range_at_r(r)
    pitch_jitter_deg = float(rng.uniform(pitch_lo, pitch_hi))
    yaw_lo, yaw_hi = yaw_range_at_r(r)
    yaw_jitter_deg = float(rng.uniform(yaw_lo, yaw_hi))
    return {
        "pos": cam,
        "forward": forward,
        "up": up,
        "r": r,
        "az": az,
        "elev": elev,
        "pitch_jitter_deg": pitch_jitter_deg,
        "yaw_jitter_deg": yaw_jitter_deg,
    }


def solve_ellipse(O: np.ndarray, C_far: np.ndarray, C_near: np.ndarray) -> Ellipse:
    """Center-at-O ellipse with C_far at the +major-axis vertex, passing through C_near.

    Plane = plane through (O, C_far, C_near). Major-axis direction û = (C_far-O)/r_far,
    semi-major a = r_far, semi-minor b chosen so the ellipse passes through C_near.
    True anomaly θ measured from C_far (so θ_far = 0, θ_near ∈ (0, π)).
    """
    O = np.asarray(O, dtype=np.float64)
    C_far = np.asarray(C_far, dtype=np.float64)
    C_near = np.asarray(C_near, dtype=np.float64)
    r_far = float(np.linalg.norm(C_far - O))
    r_near = float(np.linalg.norm(C_near - O))
    if r_far < _COLLINEAR_EPS or r_near < _COLLINEAR_EPS:
        raise DegenerateEllipse("endpoint coincides with O")
    u = (C_far - O) / r_far
    x_n = float((C_near - O).dot(u))
    y_n_sq = r_near * r_near - x_n * x_n
    if y_n_sq <= _COLLINEAR_EPS:
        raise DegenerateEllipse("collinear: C_near on the major axis line")
    y_n = float(np.sqrt(y_n_sq))
    v = ((C_near - O) - x_n * u) / y_n
    a = r_far
    denom = 1.0 - (x_n / a) ** 2
    if denom <= _DENOM_EPS:
        raise DegenerateEllipse("|x_n| ≈ a; C_near coincides with a major-axis vertex")
    b = y_n / float(np.sqrt(denom))
    theta_near = float(np.arctan2(y_n, x_n))
    return {
        "O": O,
        "u": u,
        "v": v,
        "a": a,
        "b": b,
        "theta_far": 0.0,
        "theta_near": theta_near,
    }


def ellipse_at(E: Ellipse, theta: float) -> np.ndarray:
    """Position on the ellipse at true anomaly θ (from center O)."""
    c, s = np.cos(theta), np.sin(theta)
    r = E["a"] * E["b"] / float(np.sqrt((E["b"] * c) ** 2 + (E["a"] * s) ** 2))
    return E["O"] + r * (c * E["u"] + s * E["v"])


def trajectory_thetas(E: Ellipse, n: int = N_FRAMES) -> np.ndarray:
    return np.linspace(E["theta_far"], E["theta_near"], n)


def trajectory_frames(
    E: Ellipse,
    n: int = N_FRAMES,
    pitch_start_deg: float = 0.0,
    pitch_end_deg: float = 0.0,
    yaw_start_deg: float = 0.0,
    yaw_end_deg: float = 0.0,
) -> Optional[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """N (pos, forward, up) tuples along the ellipse, look-at O, world-up reference.

    For each frame, the look-at forward/up is rotated by per-frame jitter:
        yaw_i first (around WORLD_UP), then pitch_i (around new local right).
    Frame 0 = C_far endpoint, frame n-1 = C_near endpoint.

    Returns None if any frame has a singular look-at (forward ∥ WORLD_UP).
    """
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    thetas = trajectory_thetas(E, n)
    pitch_active = abs(pitch_start_deg) > 1e-9 or abs(pitch_end_deg) > 1e-9
    yaw_active = abs(yaw_start_deg) > 1e-9 or abs(yaw_end_deg) > 1e-9
    for i, t in enumerate(thetas):
        pos = ellipse_at(E, float(t))
        look = _lookat(pos, E["O"])
        if look is None:
            return None
        forward, up = look
        alpha = (i / (n - 1)) if n > 1 else 0.0
        if yaw_active:
            yaw_i = (1.0 - alpha) * yaw_start_deg + alpha * yaw_end_deg
            forward, up = apply_yaw(forward, up, yaw_i)
        if pitch_active:
            pitch_i = (1.0 - alpha) * pitch_start_deg + alpha * pitch_end_deg
            forward, up = apply_pitch(forward, up, pitch_i)
        frames.append((pos, forward, up))
    return frames


def pose_after_jitter(pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, forward, up) with stored yaw + pitch jitter applied.

    Convention: yaw first (around WORLD_UP), then pitch (around new local right).
    """
    pos = pose["pos"]
    fwd = pose["forward"]
    up = pose["up"]
    yaw = float(pose.get("yaw_jitter_deg", 0.0))
    if abs(yaw) > 1e-9:
        fwd, up = apply_yaw(fwd, up, yaw)
    pitch = float(pose.get("pitch_jitter_deg", 0.0))
    if abs(pitch) > 1e-9:
        fwd, up = apply_pitch(fwd, up, pitch)
    return pos, fwd, up


# Backward-compat alias (older code path); to be removed once callers migrate.
def pose_after_pitch(pose: Pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pose_after_jitter(pose)


def midpoint_pose(
    E: Ellipse,
    t: float,
    pitch_far_deg: float,
    pitch_near_deg: float,
    yaw_far_deg: float = 0.0,
    yaw_near_deg: float = 0.0,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute (pos, forward, up) at fractional t ∈ [0, 1] along the ellipse arc.

    Lerps yaw + pitch between (yaw_far/pitch_far) at t=0 and (yaw_near/pitch_near)
    at t=1. Returns None on look-at singularity.
    """
    theta = (1.0 - t) * E["theta_far"] + t * E["theta_near"]
    pos = ellipse_at(E, float(theta))
    look = _lookat(pos, E["O"])
    if look is None:
        return None
    fwd, up = look
    yaw_t = (1.0 - t) * yaw_far_deg + t * yaw_near_deg
    if abs(yaw_t) > 1e-9:
        fwd, up = apply_yaw(fwd, up, yaw_t)
    pitch_t = (1.0 - t) * pitch_far_deg + t * pitch_near_deg
    if abs(pitch_t) > 1e-9:
        fwd, up = apply_pitch(fwd, up, pitch_t)
    return pos, fwd, up


def spherical_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two non-zero 3-vectors, in [0, π]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _COLLINEAR_EPS or nb < _COLLINEAR_EPS:
        return 0.0
    c = float(np.dot(a, b)) / (na * nb)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
