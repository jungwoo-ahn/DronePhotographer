"""Unit tests for src/policy/data/sampling.py — pure numpy, no Blender."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.policy.data import sampling as S


def _random_O_Cfar_Cnear(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce (O, C_far, C_near) that yield a valid (non-degenerate) ellipse."""
    while True:
        O = rng.uniform(-5, 5, size=3)
        d1 = rng.normal(size=3)
        d1 /= np.linalg.norm(d1)
        d2 = rng.normal(size=3)
        d2 /= np.linalg.norm(d2)
        if abs(d1.dot(d2)) > 0.95:
            continue
        r_far = rng.uniform(3.0, 8.0)
        r_near = rng.uniform(0.8, r_far - 0.5)
        C_far = O + r_far * d1
        C_near = O + r_near * d2
        return O, C_far, C_near


def test_worked_example_from_plan():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([10.0, 0.0, 0.0])
    C_near = np.array([3.0, 4.0, 0.0])
    E = S.solve_ellipse(O, C_far, C_near)

    assert E["a"] == pytest.approx(10.0, rel=1e-12)
    assert E["b"] == pytest.approx(4.0 / math.sqrt(0.91), rel=1e-10)
    assert E["theta_near"] == pytest.approx(math.atan2(4.0, 3.0), rel=1e-12)

    np.testing.assert_allclose(S.ellipse_at(E, 0.0), C_far, atol=1e-12)
    np.testing.assert_allclose(S.ellipse_at(E, E["theta_near"]), C_near, atol=1e-10)


def test_ellipse_passes_through_endpoints():
    rng = np.random.default_rng(0)
    for _ in range(50):
        O, C_far, C_near = _random_O_Cfar_Cnear(rng)
        E = S.solve_ellipse(O, C_far, C_near)
        np.testing.assert_allclose(S.ellipse_at(E, 0.0), C_far, atol=1e-9)
        np.testing.assert_allclose(
            S.ellipse_at(E, E["theta_near"]), C_near, atol=1e-9
        )


def test_ellipse_on_curve_normal_form():
    """For any θ, (x/a)² + (y/b)² = 1 in the local (û, v̂) frame at O."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        O, C_far, C_near = _random_O_Cfar_Cnear(rng)
        E = S.solve_ellipse(O, C_far, C_near)
        for theta in rng.uniform(0.0, 2.0 * np.pi, size=100):
            P = S.ellipse_at(E, float(theta))
            d = P - E["O"]
            x = d.dot(E["u"])
            y = d.dot(E["v"])
            z = d - x * E["u"] - y * E["v"]
            assert np.linalg.norm(z) < 1e-9, "point left the (û,v̂) plane"
            assert (x / E["a"]) ** 2 + (y / E["b"]) ** 2 == pytest.approx(
                1.0, abs=1e-9
            )


def test_degenerate_collinear_raises():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([5.0, 0.0, 0.0])
    C_near = np.array([2.0, 0.0, 0.0])
    with pytest.raises(S.DegenerateEllipse):
        S.solve_ellipse(O, C_far, C_near)


def test_degenerate_endpoint_at_O():
    O = np.array([1.0, 2.0, 3.0])
    with pytest.raises(S.DegenerateEllipse):
        S.solve_ellipse(O, O, np.array([5.0, 5.0, 5.0]))
    with pytest.raises(S.DegenerateEllipse):
        S.solve_ellipse(O, np.array([5.0, 5.0, 5.0]), O)


def test_degenerate_C_near_at_far_vertex_neighborhood():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([5.0, 0.0, 0.0])
    C_near = np.array([4.9999999, 1e-9, 0.0])
    with pytest.raises(S.DegenerateEllipse):
        S.solve_ellipse(O, C_far, C_near)


def test_log_uniform_radius_distribution():
    """Empirical CDF should match log-uniform in [R_MIN, R_MAX]."""
    rng = np.random.default_rng(42)
    O = np.array([0.0, 0.0, 0.0])
    n = 200_000
    radii = []
    for _ in range(n):
        p = S.sample_pose(O, rng)
        if p is not None:
            radii.append(p["r"])
    radii = np.array(radii)

    assert radii.min() >= S.R_MIN - 1e-9
    assert radii.max() <= S.R_MAX + 1e-9

    log_min, log_max = math.log(S.R_MIN), math.log(S.R_MAX)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        empirical = float(np.quantile(radii, q))
        expected = math.exp(log_min + q * (log_max - log_min))
        assert empirical == pytest.approx(expected, rel=0.02), (
            f"q={q}: empirical={empirical:.4f}, expected={expected:.4f}"
        )


def test_sample_pose_basis_orthonormal():
    rng = np.random.default_rng(7)
    O = np.array([1.0, -2.0, 0.5])
    for _ in range(200):
        p = S.sample_pose(O, rng)
        if p is None:
            continue
        f = p["forward"]
        u = p["up"]
        assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(f.dot(u))) < 1e-9
        assert float(f.dot(O - p["pos"])) > 0  # forward looks at O
        assert float(u.dot(S.WORLD_UP)) > 0    # up has positive world-up component


def test_trajectory_frames_endpoints_and_lookat():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([6.0, 0.0, 1.5])
    C_near = np.array([2.0, 3.0, 0.5])
    E = S.solve_ellipse(O, C_far, C_near)

    frames = S.trajectory_frames(E, n=32)
    assert frames is not None
    assert len(frames) == 32

    pos0, fwd0, up0 = frames[0]
    pos_end, fwd_end, up_end = frames[-1]
    np.testing.assert_allclose(pos0, C_far, atol=1e-9)
    np.testing.assert_allclose(pos_end, C_near, atol=1e-9)

    for pos, fwd, up in frames:
        d = O - pos
        np.testing.assert_allclose(fwd, d / np.linalg.norm(d), atol=1e-9)
        assert abs(float(fwd.dot(up))) < 1e-9


def test_trajectory_singularity_returns_none(monkeypatch):
    """A trajectory whose first frame has forward ∥ world_up returns None."""
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([0.0, 0.0, 5.0])         # straight above O
    C_near = np.array([3.0, 0.0, 0.001])
    E = S.solve_ellipse(O, C_far, C_near)
    frames = S.trajectory_frames(E, n=8)
    assert frames is None


def test_spherical_angle_symmetry_and_range():
    rng = np.random.default_rng(11)
    for _ in range(100):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        ang = S.spherical_angle(a, b)
        assert 0.0 <= ang <= math.pi + 1e-12
        assert ang == pytest.approx(S.spherical_angle(b, a), abs=1e-12)
        assert S.spherical_angle(a, a) == pytest.approx(0.0, abs=1e-6)
        assert S.spherical_angle(a, -a) == pytest.approx(math.pi, abs=1e-6)


def test_spherical_angle_zero_vector_returns_zero():
    assert S.spherical_angle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.0


def test_pitch_range_at_r_endpoints_and_midpoint():
    r_n, lo_n, hi_n = S.PITCH_LERP_NEAR
    r_f, lo_f, hi_f = S.PITCH_LERP_FAR
    assert S.pitch_range_at_r(r_n) == pytest.approx((lo_n, hi_n))
    assert S.pitch_range_at_r(r_f) == pytest.approx((lo_f, hi_f))
    mid_r = (r_n + r_f) / 2.0
    assert S.pitch_range_at_r(mid_r) == pytest.approx(
        ((lo_n + lo_f) / 2.0, (hi_n + hi_f) / 2.0)
    )


def test_pitch_range_at_r_clips_outside_envelope():
    _, lo_n, hi_n = S.PITCH_LERP_NEAR
    _, lo_f, hi_f = S.PITCH_LERP_FAR
    # Below NEAR clamps to NEAR.
    assert S.pitch_range_at_r(0.1) == pytest.approx((lo_n, hi_n))
    # Above FAR clamps to FAR.
    assert S.pitch_range_at_r(100.0) == pytest.approx((lo_f, hi_f))


def test_apply_pitch_zero_is_identity():
    fwd = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    f2, u2 = S.apply_pitch(fwd, up, 0.0)
    np.testing.assert_allclose(f2, fwd, atol=1e-12)
    np.testing.assert_allclose(u2, up, atol=1e-12)


def test_apply_pitch_preserves_orthonormality():
    rng = np.random.default_rng(13)
    for _ in range(50):
        # Build a random forward not parallel to WORLD_UP.
        while True:
            d = rng.normal(size=3)
            if abs(d[2]) < 0.95 * np.linalg.norm(d):
                break
        fwd = d / np.linalg.norm(d)
        # build up from world_up orthogonalized
        up_raw = S.WORLD_UP - S.WORLD_UP.dot(fwd) * fwd
        up = up_raw / np.linalg.norm(up_raw)
        pitch_deg = float(rng.uniform(-60.0, 60.0))
        f2, u2 = S.apply_pitch(fwd, up, pitch_deg)
        assert np.linalg.norm(f2) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(u2) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(f2.dot(u2))) < 1e-9
        # Rotation around right axis preserves the right axis direction (sign may flip
        # depending on hemisphere; we check the right vector is consistent).
        right_orig = np.cross(fwd, S.WORLD_UP)
        right_orig = right_orig / np.linalg.norm(right_orig)
        right_new = np.cross(f2, u2)
        # right_new should be ± right_orig.
        assert abs(abs(float(right_new.dot(right_orig))) - 1.0) < 1e-9


def test_apply_pitch_known_angle():
    # Camera at +Y axis looking at origin: forward = -Y, up = +Z, right = forward × up = -X.
    # Wait: right = forward × world_up = (-Y) × (+Z) = +X. Good.
    # Pitch +90° around +X rotates forward from -Y to +Z (looking straight up).
    fwd = np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    f2, u2 = S.apply_pitch(fwd, up, 90.0)
    np.testing.assert_allclose(f2, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_sample_pose_returns_pitch_in_range():
    rng = np.random.default_rng(99)
    O = np.array([0.0, 0.0, 0.0])
    seen = 0
    for _ in range(200):
        p = S.sample_pose(O, rng)
        if p is None:
            continue
        lo, hi = S.pitch_range_at_r(p["r"])
        assert lo - 1e-9 <= p["pitch_jitter_deg"] <= hi + 1e-9
        seen += 1
    assert seen > 50


def test_yaw_range_at_r_endpoints_and_clip():
    r_n, lo_n, hi_n = S.YAW_LERP_NEAR
    r_f, lo_f, hi_f = S.YAW_LERP_FAR
    assert S.yaw_range_at_r(r_n) == pytest.approx((lo_n, hi_n))
    assert S.yaw_range_at_r(r_f) == pytest.approx((lo_f, hi_f))
    mid = (r_n + r_f) / 2.0
    assert S.yaw_range_at_r(mid) == pytest.approx(
        ((lo_n + lo_f) / 2.0, (hi_n + hi_f) / 2.0)
    )
    assert S.yaw_range_at_r(0.1) == pytest.approx((lo_n, hi_n))
    assert S.yaw_range_at_r(100.0) == pytest.approx((lo_f, hi_f))


def test_apply_yaw_zero_is_identity():
    fwd = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    f2, u2 = S.apply_yaw(fwd, up, 0.0)
    np.testing.assert_allclose(f2, fwd, atol=1e-12)
    np.testing.assert_allclose(u2, up, atol=1e-12)


def test_apply_yaw_known_angle():
    # forward = +Y. Yaw +90° around +Z rotates +Y to -X.
    fwd = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    f2, u2 = S.apply_yaw(fwd, up, 90.0)
    np.testing.assert_allclose(f2, np.array([-1.0, 0.0, 0.0]), atol=1e-9)
    # Up should be preserved if it's parallel to WORLD_UP.
    np.testing.assert_allclose(u2, up, atol=1e-9)


def test_apply_yaw_preserves_orthonormality():
    rng = np.random.default_rng(17)
    for _ in range(50):
        d = rng.normal(size=3)
        if abs(d[2]) > 0.95 * np.linalg.norm(d):
            continue
        fwd = d / np.linalg.norm(d)
        up_raw = S.WORLD_UP - S.WORLD_UP.dot(fwd) * fwd
        up = up_raw / np.linalg.norm(up_raw)
        yaw_deg = float(rng.uniform(-60.0, 60.0))
        f2, u2 = S.apply_yaw(fwd, up, yaw_deg)
        assert np.linalg.norm(f2) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(u2) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(f2.dot(u2))) < 1e-9


def test_sample_pose_returns_yaw_in_range():
    rng = np.random.default_rng(31)
    O = np.array([0.0, 0.0, 0.0])
    seen = 0
    for _ in range(200):
        p = S.sample_pose(O, rng)
        if p is None:
            continue
        lo, hi = S.yaw_range_at_r(p["r"])
        assert lo - 1e-9 <= p["yaw_jitter_deg"] <= hi + 1e-9
        seen += 1
    assert seen > 50


def test_trajectory_frames_yaw_lerp_endpoints():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([5.0, 0.0, 1.0])
    C_near = np.array([2.0, 3.0, 0.5])
    E = S.solve_ellipse(O, C_far, C_near)
    yaw_s, yaw_e = +25.0, -15.0
    frames_jit = S.trajectory_frames(
        E, n=8, yaw_start_deg=yaw_s, yaw_end_deg=yaw_e
    )
    frames_raw = S.trajectory_frames(E, n=8)
    assert frames_jit is not None and frames_raw is not None

    def horiz_angle(a, b):
        # angle between forward vectors projected onto the xy plane (around z)
        a2 = np.array([a[0], a[1], 0.0])
        b2 = np.array([b[0], b[1], 0.0])
        if np.linalg.norm(a2) < 1e-9 or np.linalg.norm(b2) < 1e-9:
            return 0.0
        a2 /= np.linalg.norm(a2)
        b2 /= np.linalg.norm(b2)
        return float(np.degrees(np.arccos(np.clip(a2.dot(b2), -1.0, 1.0))))

    a0 = horiz_angle(frames_jit[0][1], frames_raw[0][1])
    aN = horiz_angle(frames_jit[-1][1], frames_raw[-1][1])
    assert a0 == pytest.approx(abs(yaw_s), abs=1e-6)
    assert aN == pytest.approx(abs(yaw_e), abs=1e-6)


def test_pose_after_jitter_applies_both():
    O = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(42)
    p = None
    for _ in range(30):
        cand = S.sample_pose(O, rng)
        if cand is not None and abs(cand["yaw_jitter_deg"]) > 1.0 and abs(cand["pitch_jitter_deg"]) > 1.0:
            p = cand
            break
    assert p is not None
    pos, fwd, up = S.pose_after_jitter(p)
    # Position unchanged.
    np.testing.assert_allclose(pos, p["pos"], atol=1e-12)
    # Forward differs from look-at forward (since jitter applied).
    assert not np.allclose(fwd, p["forward"], atol=1e-3)
    # Still unit + orthogonal.
    assert np.linalg.norm(fwd) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-9)
    assert abs(float(fwd.dot(up))) < 1e-9


def test_trajectory_frames_pitch_lerp_endpoints():
    O = np.array([0.0, 0.0, 0.0])
    C_far = np.array([5.0, 0.0, 1.0])
    C_near = np.array([2.0, 3.0, 0.5])
    E = S.solve_ellipse(O, C_far, C_near)
    pitch_s, pitch_e = +20.0, -10.0
    frames_jit = S.trajectory_frames(E, n=8, pitch_start_deg=pitch_s, pitch_end_deg=pitch_e)
    frames_raw = S.trajectory_frames(E, n=8)
    assert frames_jit is not None and frames_raw is not None
    # Frame 0: pitch ≈ pitch_s. Frame -1: pitch ≈ pitch_e.
    # Check by comparing forward vectors against the raw look-at counterpart.
    def angle_deg(a, b):
        return float(np.degrees(np.arccos(np.clip(a.dot(b), -1.0, 1.0))))
    a0 = angle_deg(frames_jit[0][1], frames_raw[0][1])
    aN = angle_deg(frames_jit[-1][1], frames_raw[-1][1])
    assert a0 == pytest.approx(abs(pitch_s), abs=1e-6)
    assert aN == pytest.approx(abs(pitch_e), abs=1e-6)
