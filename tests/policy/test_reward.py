"""Unit tests for src/policy/common/reward.py."""

import math

import numpy as np
import pytest

from src.policy.common.reward import (
    CameraIntrinsics,
    geometric_profile_distance,
    pose_distance_value,
    pose_to_geometry,
    profile_distance_value,
    profile_to_geometry,
    score_distance,
    score_distance_reward,
)

INTR = CameraIntrinsics.from_render(1024, 768)


def _profile(**kw):
    base = dict(occupancy=30, body_in_frame_ratio=100, cam_to_obj_azimuth_deg=180,
                cam_to_obj_elevation_deg=0, object_center_x=512, object_center_y=384,
                bbox_x_offset=150, bbox_y_offset=300)
    base.update(kw)
    return base


def test_geometric_distance_zero_at_identity():
    p = _profile()
    assert geometric_profile_distance(p, p, INTR) == 0.0
    assert profile_distance_value(p, p, INTR) == 0.0


def test_value_is_nonpositive():
    assert profile_distance_value(_profile(cam_to_obj_azimuth_deg=90), _profile(), INTR) <= 0.0


def test_azimuth_is_cyclic():
    # 350° vs 10° is 20° apart, not 340°
    near = geometric_profile_distance(_profile(cam_to_obj_azimuth_deg=350),
                                      _profile(cam_to_obj_azimuth_deg=10), INTR)
    assert math.degrees(near) == pytest.approx(20.0, abs=1.0)


def test_viewing_distance_monotonic():
    g = _profile(cam_to_obj_azimuth_deg=180)
    d30 = geometric_profile_distance(_profile(cam_to_obj_azimuth_deg=210), g, INTR)
    d90 = geometric_profile_distance(_profile(cam_to_obj_azimuth_deg=270), g, INTR)
    assert 0 < d30 < d90


def test_polar_degeneracy_azimuth_matters_less_near_poles():
    # Same 40° azimuth gap contributes less to the distance at high elevation.
    at_equator = geometric_profile_distance(
        _profile(cam_to_obj_azimuth_deg=0, cam_to_obj_elevation_deg=0),
        _profile(cam_to_obj_azimuth_deg=40, cam_to_obj_elevation_deg=0), INTR)
    near_pole = geometric_profile_distance(
        _profile(cam_to_obj_azimuth_deg=0, cam_to_obj_elevation_deg=80),
        _profile(cam_to_obj_azimuth_deg=40, cam_to_obj_elevation_deg=80), INTR)
    assert near_pole < at_equator


def test_size_term_contributes():
    g = _profile(bbox_y_offset=200)
    d = geometric_profile_distance(_profile(bbox_y_offset=600), g, INTR)
    assert d > 0


def test_aim_term_contributes():
    g = _profile(object_center_x=512)
    d = geometric_profile_distance(_profile(object_center_x=900), g, INTR)
    assert d > 0


def test_offframe_aim_is_bounded():
    # A subject far off-frame must not blow up — atan caps the aim angle below pi/2.
    geom = profile_to_geometry(_profile(object_center_x=100000), INTR)
    assert abs(geom["aim_x"]) < math.pi / 2


def test_offframe_size_is_bounded():
    geom = profile_to_geometry(_profile(bbox_y_offset=100000), INTR)
    assert geom["size"] < math.pi / 2


# --- pose-based geometry (training-time value path) ---------------------------

CENTER = [0.0, 0.0, 1.0]
HEIGHT = 1.8


def test_pose_geometry_camera_looking_straight_at_subject():
    # camera 3 m away on +x, looking back along -x, up = +z
    g = pose_to_geometry([3, 0, 1], [-1, 0, 0], [0, 0, 1], CENTER, HEIGHT)
    assert g["aim_x"] == pytest.approx(0.0, abs=1e-6)
    assert g["aim_y"] == pytest.approx(0.0, abs=1e-6)
    assert g["el"] == pytest.approx(0.0, abs=1e-6)
    assert g["size"] == pytest.approx(math.atan(0.9 / 3.0), abs=1e-6)


def test_pose_geometry_camera_above_gives_negative_elevation():
    # v2 convention: camera above the subject -> elevation negative
    g = pose_to_geometry([0, 0, 4], [0, 0, -1], [0, 1, 0], CENTER, HEIGHT)
    assert g["el"] < -1.0  # near -pi/2


def test_pose_geometry_aim_sign_conventions():
    # camera at +x looking -x with up +z: camera-right is +y.
    # subject offset to +y => to the camera's right => aim_x > 0
    g = pose_to_geometry([3, 0, 1], [-1, 0, 0], [0, 0, 1], [0, 0.5, 1], HEIGHT)
    assert g["aim_x"] > 0
    # subject above the optical axis => aim_y < 0 (image-y points down)
    g2 = pose_to_geometry([3, 0, 1], [-1, 0, 0], [0, 0, 1], [0, 0, 1.5], HEIGHT)
    assert g2["aim_y"] < 0


def test_pose_value_zero_at_same_pose():
    v = pose_distance_value([3, 0, 1], [-1, 0, 0], [0, 0, 1],
                            [3, 0, 1], [-1, 0, 0], [0, 0, 1],
                            subject_center=CENTER, subject_height=HEIGHT)
    assert v == pytest.approx(0.0, abs=1e-6)


def test_pose_value_negative_and_monotone_in_orbit_angle():
    import numpy as np
    goal_th = 0.0
    def pose(th):
        pos = [3*math.cos(th), 3*math.sin(th), 1.0]
        fwd = [-math.cos(th), -math.sin(th), 0.0]
        return pos, fwd, [0.0, 0.0, 1.0]
    gp, gf, gu = pose(goal_th)
    vals = []
    for th in [0.2, 0.6, 1.2, 2.0]:
        p, f, u = pose(th)
        vals.append(pose_distance_value(p, f, u, gp, gf, gu,
                                        subject_center=CENTER, subject_height=HEIGHT))
    assert all(v < 0 for v in vals)
    # farther around the orbit => more negative
    assert vals == sorted(vals, reverse=True)


def test_pose_value_immune_to_close_range():
    # Very close camera: the bbox projection would blow up and clamp, but the
    # pose-based value stays finite and sane.
    v = pose_distance_value([0.3, 0, 1], [-1, 0, 0], [0, 0, 1],
                            [3, 0, 1], [-1, 0, 0], [0, 0, 1],
                            subject_center=CENTER, subject_height=HEIGHT)
    assert math.isfinite(v)
    assert -math.pi < v < 0


def test_zero_distance_when_equal():
    keys = ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"]
    a = np.array([45.0, 10.0], dtype=np.float32)
    assert score_distance(a, a, keys) == 0.0
    assert score_distance_reward(a, a, keys) == 0.0


def test_distance_grows_with_difference():
    keys = ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"]
    a = np.array([45.0, 10.0], dtype=np.float32)
    b = np.array([60.0, 10.0], dtype=np.float32)
    c = np.array([120.0, 10.0], dtype=np.float32)
    d_ab = score_distance(a, b, keys)
    d_ac = score_distance(a, c, keys)
    assert 0 < d_ab < d_ac


def test_reward_is_negative_of_distance():
    keys = ["cam_to_obj_azimuth_deg"]
    a = np.array([0.0], dtype=np.float32)
    b = np.array([90.0], dtype=np.float32)
    d = score_distance(a, b, keys)
    r = score_distance_reward(a, b, keys)
    assert r == -d


def test_weighted_distance_emphasizes_high_weight_keys():
    keys = ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"]
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([45.0, 0.0], dtype=np.float32)
    d_low_w = score_distance(a, b, keys, weights={"cam_to_obj_azimuth_deg": 0.1, "cam_to_obj_elevation_deg": 1.0})
    d_high_w = score_distance(a, b, keys, weights={"cam_to_obj_azimuth_deg": 1.0, "cam_to_obj_elevation_deg": 0.1})
    assert d_high_w > d_low_w
