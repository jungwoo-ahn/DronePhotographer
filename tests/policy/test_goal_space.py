"""Unit tests for src/policy/common/goal_space.py."""

import numpy as np

from src.policy.common.goal_space import (
    DEFAULT_GOAL_KEYS,
    DEFAULT_V5_RANGES,
    denormalize_goal,
    derive_partial_v5_from_final_image,
    goal_keys,
    goal_vector,
    has_all_keys,
    normalize_goal,
)


def test_default_keys_match_v5():
    assert "occupancy" in DEFAULT_GOAL_KEYS
    assert "cam_to_obj_azimuth_deg" in DEFAULT_GOAL_KEYS
    assert "cam_to_obj_elevation_deg" in DEFAULT_GOAL_KEYS
    assert len(DEFAULT_GOAL_KEYS) == 8


def test_goal_vector_picks_up_bare_keys():
    ann = {"occupancy": 50.0, "cam_to_obj_azimuth_deg": -90.0}
    keys = ["occupancy", "cam_to_obj_azimuth_deg"]
    v = goal_vector(ann, keys)
    assert v.shape == (2,)
    assert v.dtype == np.float32
    np.testing.assert_allclose(v, [50.0, -90.0])


def test_goal_vector_picks_up_score_prefixed_keys():
    ann = {"score_occupancy": 80.0}
    v = goal_vector(ann, ["occupancy"])
    assert v[0] == 80.0


def test_goal_vector_missing_keys_become_nan():
    v = goal_vector({}, ["occupancy"])
    assert np.isnan(v[0])


def test_derive_partial_v5_from_v6_final_image():
    fi = {"azimuth": 45.0, "elevation": 10.0, "camera_position": [0, 0, 0]}
    out = derive_partial_v5_from_final_image(fi)
    assert out == {"cam_to_obj_azimuth_deg": 45.0, "cam_to_obj_elevation_deg": 10.0}


def test_goal_vector_falls_back_to_derived_v5_for_v6():
    # v6 placements expose azimuth/elevation but not the bare V5 keys
    fi = {"azimuth": 45.0, "elevation": 10.0}
    v = goal_vector(fi, ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"])
    np.testing.assert_allclose(v, [45.0, 10.0])


def test_normalize_round_trip():
    keys = ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"]
    original = np.array([90.0, -45.0], dtype=np.float32)
    n = normalize_goal(original, keys)
    # az ∈ (0,360): 90 → 2*90/360-1 = -0.5; el ∈ (-90,90): -45 → -0.5
    np.testing.assert_allclose(n, [-0.5, -0.5], atol=1e-6)
    back = denormalize_goal(n, keys)
    np.testing.assert_allclose(back, original, atol=1e-4)


def test_azimuth_full_range_maps_to_unit_interval():
    keys = ["cam_to_obj_azimuth_deg"]
    # 0 -> -1, 180 -> 0, 360 -> +1
    np.testing.assert_allclose(normalize_goal(np.array([0.0], np.float32), keys), [-1.0], atol=1e-6)
    np.testing.assert_allclose(normalize_goal(np.array([180.0], np.float32), keys), [0.0], atol=1e-6)
    np.testing.assert_allclose(normalize_goal(np.array([360.0], np.float32), keys), [1.0], atol=1e-6)
    # 243 (the handoff example) is in-range, no longer saturates
    assert -1.0 < float(normalize_goal(np.array([243.0], np.float32), keys)[0]) < 1.0


def test_pixel_keys_normalize_against_render_resolution():
    from src.policy.common.goal_space import RENDER_WIDTH, RENDER_HEIGHT
    # object_center_x = 419 px on a 1024-wide render → in-range, not saturated
    nx = normalize_goal(np.array([419.0], np.float32), ["object_center_x"])
    assert -1.0 < float(nx[0]) < 1.0
    # center of frame maps to 0
    np.testing.assert_allclose(
        normalize_goal(np.array([RENDER_WIDTH / 2], np.float32), ["object_center_x"]), [0.0], atol=1e-6,
    )
    np.testing.assert_allclose(
        normalize_goal(np.array([RENDER_HEIGHT / 2], np.float32), ["object_center_y"]), [0.0], atol=1e-6,
    )


def test_normalize_clips_out_of_range():
    keys = ["cam_to_obj_azimuth_deg"]
    huge = np.array([10000.0], dtype=np.float32)
    n = normalize_goal(huge, keys)
    assert n[0] == 1.0


def test_has_all_keys():
    fi = {"azimuth": 45.0, "elevation": 10.0}
    assert has_all_keys(fi, ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"])
    assert not has_all_keys(fi, ["occupancy"])


def test_default_v5_ranges_cover_all_keys():
    for k in DEFAULT_GOAL_KEYS:
        assert k in DEFAULT_V5_RANGES, f"missing range for {k}"


def test_custom_keys_override_default():
    keys = goal_keys(["occupancy"])
    assert keys == ["occupancy"]
    keys2 = goal_keys()
    assert keys2 == DEFAULT_GOAL_KEYS
