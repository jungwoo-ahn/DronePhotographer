"""Tests for _describe_targets in src/vlm_qwen25/mpc.py."""
from __future__ import annotations

import pytest

from src.vlm_qwen25.mpc import _describe_targets


def _make(targets: dict[str, float], weights: dict[str, float] | None = None) -> str:
    if weights is None:
        weights = {k: 1.0 for k in targets}
    return _describe_targets(targets, weights)


class TestDescribeTargets:
    def test_front_eyelevel(self):
        desc = _make({"camera_to_object_fy": 1.0, "camera_to_object_fz": 0.0})
        assert "front" in desc
        assert "eye-level" in desc

    def test_back_from_above(self):
        desc = _make({"camera_to_object_fy": -1.0, "camera_to_object_fz": -0.5})
        assert "back" in desc
        assert "from above" in desc

    def test_left_side(self):
        desc = _make({
            "camera_to_object_fx": 1.0,
            "camera_to_object_fy": 0.0,
            "camera_to_object_fz": 0.0,
        })
        assert "left side" in desc

    def test_right_front(self):
        desc = _make({
            "camera_to_object_fx": -0.7,
            "camera_to_object_fy": 0.7,
            "camera_to_object_fz": 0.0,
        })
        assert "right-front" in desc

    def test_topdown(self):
        desc = _make({"camera_to_object_uy": -1.0})
        assert "top-down" in desc

    def test_from_below(self):
        desc = _make({"camera_to_object_fz": 0.5})
        assert "from below" in desc

    def test_size_percentage(self):
        desc = _make({"bbox_occupancy_ratio": 0.4})
        assert "size 40%" in desc

    def test_centered(self):
        desc = _make({"bbox_centroid_offset": 0.0})
        assert "centered" in desc

    def test_not_centered(self):
        desc = _make({"bbox_centroid_offset": 0.1})
        assert "centered" not in desc

    def test_zero_weight_excluded(self):
        desc = _make(
            {"camera_to_object_fy": 1.0},
            weights={"camera_to_object_fy": 0.0},
        )
        assert desc == ""

    def test_empty(self):
        desc = _make({})
        assert desc == ""

    def test_full_combination(self):
        desc = _make({
            "bbox_occupancy_ratio": 0.5,
            "bbox_centroid_offset": 0.0,
            "camera_to_object_fx": -0.5,
            "camera_to_object_fy": 0.8,
            "camera_to_object_fz": -0.3,
        })
        assert "size 50%" in desc
        assert "centered" in desc
        assert "right-front" in desc
        assert "from above" in desc

    def test_topdown_overrides_fz(self):
        desc = _make({
            "camera_to_object_uy": -0.8,
            "camera_to_object_fz": -0.3,
        })
        assert "top-down" in desc
        assert "from above" not in desc
