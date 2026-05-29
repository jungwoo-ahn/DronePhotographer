"""Unit tests for src/policy/common/reward.py."""

import numpy as np

from src.policy.common.reward import score_distance, score_distance_reward


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
