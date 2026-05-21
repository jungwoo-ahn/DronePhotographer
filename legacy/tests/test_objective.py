"""Tests for src/vlm_qwen25/objective.py."""
from __future__ import annotations

import math

import pytest

from src.vlm_qwen25.objective import (
    build_target_objective,
    compile_score_targets,
    compile_score_weights,
    composition_to_margin_targets,
    weighted_target_error,
    DEFAULT_SCORE_WEIGHTS,
    TARGET_PRESETS,
    TargetObjective,
)


# ---------------------------------------------------------------------------
# composition_to_margin_targets
# ---------------------------------------------------------------------------

class TestCompositionToMarginTargets:
    def test_centered_square(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.25, 1.0, 1.0)
        # width = height = sqrt(0.25) = 0.5, all margins = 0.25
        assert m["bbox_margin_top"] == pytest.approx(0.25, abs=1e-6)
        assert m["bbox_margin_bottom"] == pytest.approx(0.25, abs=1e-6)
        assert m["bbox_margin_left"] == pytest.approx(0.25, abs=1e-6)
        assert m["bbox_margin_right"] == pytest.approx(0.25, abs=1e-6)

    def test_centered_with_frame_ar(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.25, 1.0, 4.0 / 3.0)
        # Symmetric: left == right, top == bottom
        assert m["bbox_margin_left"] == pytest.approx(m["bbox_margin_right"], abs=1e-6)
        assert m["bbox_margin_top"] == pytest.approx(m["bbox_margin_bottom"], abs=1e-6)

    def test_off_center(self):
        m = composition_to_margin_targets(0.667, 0.333, 0.25, 1.0, 4.0 / 3.0)
        assert m["bbox_margin_left"] > m["bbox_margin_right"]
        assert m["bbox_margin_top"] < m["bbox_margin_bottom"]

    def test_algebraic_invariant(self):
        """left + right + width = 1.0, top + bottom + height = 1.0."""
        cx, cy, occ, ar, far = 0.6, 0.4, 0.3, 1.2, 4.0 / 3.0
        m = composition_to_margin_targets(cx, cy, occ, ar, far)
        norm_ar = ar / far
        width = math.sqrt(occ * norm_ar)
        height = math.sqrt(occ / norm_ar)
        assert m["bbox_margin_left"] + m["bbox_margin_right"] + width == pytest.approx(1.0, abs=1e-6)
        assert m["bbox_margin_top"] + m["bbox_margin_bottom"] + height == pytest.approx(1.0, abs=1e-6)

    def test_full_occupancy(self):
        m = composition_to_margin_targets(0.5, 0.5, 1.0, 1.0, 1.0)
        for key in m:
            assert m[key] == pytest.approx(0.0, abs=1e-6)

    def test_tiny_occupancy(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.01, 1.0, 1.0)
        # width = height = 0.1
        for key in m:
            assert m[key] == pytest.approx(0.45, abs=1e-6)

    def test_portrait_aspect(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.3, 0.67, 4.0 / 3.0)
        norm_ar = 0.67 / (4.0 / 3.0)
        width = math.sqrt(0.3 * norm_ar)
        height = math.sqrt(0.3 / norm_ar)
        assert height > width  # portrait in landscape frame

    def test_wide_aspect(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.3, 1.8, 4.0 / 3.0)
        norm_ar = 1.8 / (4.0 / 3.0)
        width = math.sqrt(0.3 * norm_ar)
        height = math.sqrt(0.3 / norm_ar)
        assert width > height

    def test_occupancy_over_one_raises(self):
        with pytest.raises(ValueError, match="occupancy must be in"):
            composition_to_margin_targets(0.5, 0.5, 1.5, 1.0, 1.0)

    def test_negative_occupancy_raises(self):
        with pytest.raises(ValueError, match="occupancy must be in"):
            composition_to_margin_targets(0.5, 0.5, -0.1, 1.0, 1.0)

    def test_zero_aspect_ratio_raises(self):
        with pytest.raises(ValueError, match="aspect_ratio must be > 0"):
            composition_to_margin_targets(0.5, 0.5, 0.3, 0.0, 1.0)

    def test_negative_frame_ar_raises(self):
        with pytest.raises(ValueError, match="frame_aspect_ratio must be > 0"):
            composition_to_margin_targets(0.5, 0.5, 0.3, 1.0, -1.0)

    def test_center_out_of_range_raises(self):
        with pytest.raises(ValueError, match="center_x and center_y must be in"):
            composition_to_margin_targets(1.5, 0.5, 0.1, 1.0, 1.0)

    def test_does_not_fit_raises(self):
        with pytest.raises(ValueError, match="does not fit inside the frame"):
            composition_to_margin_targets(0.9, 0.9, 0.5, 1.0, 1.0)

    def test_margins_non_negative(self):
        m = composition_to_margin_targets(0.5, 0.5, 0.99, 1.0, 1.0)
        for key, value in m.items():
            assert value >= 0.0, f"{key} is negative: {value}"


# ---------------------------------------------------------------------------
# compile_score_targets
# ---------------------------------------------------------------------------

class TestCompileScoreTargets:
    def test_composition_centered(self):
        spec = {"center_x": 0.5, "center_y": 0.5, "occupancy": 0.3, "aspect_ratio": 1.0}
        targets = compile_score_targets(spec)
        assert "bbox_occupancy_ratio" in targets
        assert "bbox_aspect_ratio" in targets
        assert "bbox_centroid_offset" in targets
        assert targets["bbox_centroid_offset"] == 0.0
        for margin_key in ["bbox_margin_top", "bbox_margin_bottom", "bbox_margin_left", "bbox_margin_right"]:
            assert margin_key in targets

    def test_composition_off_center(self):
        spec = {"center_x": 0.667, "center_y": 0.333, "occupancy": 0.25, "aspect_ratio": 1.0}
        targets = compile_score_targets(spec)
        assert "bbox_centroid_offset" not in targets  # not centered

    def test_direct_bbox_keys(self):
        spec = {"bbox_occupancy_ratio": 0.5, "bbox_centroid_offset": 0.0}
        targets = compile_score_targets(spec)
        assert targets == {"bbox_occupancy_ratio": 0.5, "bbox_centroid_offset": 0.0}

    def test_c2o_keys_passthrough(self):
        spec = {
            "camera_to_object_fx": 0.0,
            "camera_to_object_fy": 1.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 0.0,
            "camera_to_object_uz": 1.0,
        }
        targets = compile_score_targets(spec)
        assert len(targets) == 6
        assert targets["camera_to_object_fy"] == 1.0

    def test_mixed_composition_and_c2o(self):
        spec = {
            "center_x": 0.5, "center_y": 0.5,
            "occupancy": 0.4, "aspect_ratio": 1.0,
            "camera_to_object_fx": 0.0, "camera_to_object_fy": 1.0,
            "camera_to_object_fz": 0.0, "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 0.0, "camera_to_object_uz": 1.0,
        }
        targets = compile_score_targets(spec)
        assert "bbox_margin_top" in targets
        assert "camera_to_object_fy" in targets

    def test_center_x_without_center_y_raises(self):
        with pytest.raises(ValueError, match="center_x and center_y must be provided together"):
            compile_score_targets({"center_x": 0.5, "occupancy": 0.3})

    def test_off_center_without_occupancy_raises(self):
        with pytest.raises(ValueError, match="off-center composition targets require"):
            compile_score_targets({"center_x": 0.7, "center_y": 0.3})

    def test_centered_without_occupancy(self):
        targets = compile_score_targets({"center_x": 0.5, "center_y": 0.5})
        assert targets == {"bbox_centroid_offset": 0.0}

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="target spec did not produce any score targets"):
            compile_score_targets({})

    def test_occupancy_shorthand(self):
        targets = compile_score_targets({"occupancy": 0.4})
        assert targets["bbox_occupancy_ratio"] == 0.4

    def test_aspect_ratio_shorthand(self):
        targets = compile_score_targets({"aspect_ratio": 1.5})
        assert targets["bbox_aspect_ratio"] == 1.5

    def test_unknown_keys_ignored(self):
        targets = compile_score_targets({"occupancy": 0.3, "unknown_key": 42.0})
        assert "unknown_key" not in targets
        assert "bbox_occupancy_ratio" in targets

    def test_all_presets_compile(self):
        """Every TARGET_PRESET must produce valid score_targets."""
        for name, spec in TARGET_PRESETS.items():
            targets = compile_score_targets(spec)
            assert len(targets) > 0, f"preset '{name}' produced empty targets"


# ---------------------------------------------------------------------------
# compile_score_weights
# ---------------------------------------------------------------------------

class TestCompileScoreWeights:
    def test_defaults(self):
        targets = {"bbox_occupancy_ratio": 0.5, "bbox_centroid_offset": 0.0}
        weights = compile_score_weights(targets)
        assert weights["bbox_occupancy_ratio"] == 2.0
        assert weights["bbox_centroid_offset"] == 2.0

    def test_custom_override(self):
        targets = {"bbox_occupancy_ratio": 0.5}
        weights = compile_score_weights(targets, {"bbox_occupancy_ratio": 5.0})
        assert weights["bbox_occupancy_ratio"] == 5.0

    def test_c2o_fallback_to_one(self):
        targets = {"camera_to_object_fx": 0.0}
        weights = compile_score_weights(targets)
        assert weights["camera_to_object_fx"] == 1.0

    def test_negative_weight_raises(self):
        targets = {"bbox_occupancy_ratio": 0.5}
        with pytest.raises(ValueError, match="score weight must be non-negative"):
            compile_score_weights(targets, {"bbox_occupancy_ratio": -1.0})

    def test_zero_weight_allowed(self):
        targets = {"bbox_occupancy_ratio": 0.5}
        weights = compile_score_weights(targets, {"bbox_occupancy_ratio": 0.0})
        assert weights["bbox_occupancy_ratio"] == 0.0

    def test_only_target_keys_in_result(self):
        targets = {"bbox_occupancy_ratio": 0.5}
        weights = compile_score_weights(targets, {"bbox_occupancy_ratio": 1.0, "extra_key": 99.0})
        assert "extra_key" not in weights
        assert len(weights) == 1


# ---------------------------------------------------------------------------
# weighted_target_error
# ---------------------------------------------------------------------------

class TestWeightedTargetError:
    def test_perfect_match(self):
        targets = {"a": 0.5, "b": 0.3}
        predicted = {"a": 0.5, "b": 0.3}
        weights = {"a": 1.0, "b": 1.0}
        error, per_key = weighted_target_error(predicted, targets, weights)
        assert error == pytest.approx(0.0)
        assert per_key["a"] == pytest.approx(0.0)
        assert per_key["b"] == pytest.approx(0.0)

    def test_single_key_error(self):
        error, per_key = weighted_target_error({"a": 0.3}, {"a": 0.5}, {"a": 1.0})
        assert error == pytest.approx(0.2)
        assert per_key["a"] == pytest.approx(0.2)

    def test_weighted_multi_key(self):
        predicted = {"a": 0.0, "b": 0.0}
        targets = {"a": 1.0, "b": 1.0}
        weights = {"a": 2.0, "b": 1.0}
        error, per_key = weighted_target_error(predicted, targets, weights)
        # (2*1 + 1*1) / (2+1) = 1.0
        assert error == pytest.approx(1.0)

    def test_zero_weight_skipped(self):
        predicted = {"a": 0.0, "b": 0.0}
        targets = {"a": 1.0, "b": 1.0}
        weights = {"a": 0.0, "b": 1.0}
        error, per_key = weighted_target_error(predicted, targets, weights)
        assert error == pytest.approx(1.0)
        assert "a" not in per_key  # skipped

    def test_all_zero_weights(self):
        predicted = {"a": 0.0}
        targets = {"a": 1.0}
        weights = {"a": 0.0}
        error, per_key = weighted_target_error(predicted, targets, weights)
        assert error == 0.0

    def test_missing_weight_defaults_to_one(self):
        predicted = {"a": 0.0}
        targets = {"a": 1.0}
        weights = {}  # missing 'a'
        error, per_key = weighted_target_error(predicted, targets, weights)
        assert error == pytest.approx(1.0)

    def test_negative_predicted(self):
        # c2o values can be negative
        predicted = {"a": -0.5}
        targets = {"a": 0.5}
        weights = {"a": 1.0}
        error, per_key = weighted_target_error(predicted, targets, weights)
        assert error == pytest.approx(1.0)
        assert per_key["a"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_target_objective
# ---------------------------------------------------------------------------

class TestBuildTargetObjective:
    def test_from_preset(self):
        obj = build_target_objective(preset_name="centered_50")
        assert isinstance(obj, TargetObjective)
        assert obj.name == "centered_50"
        assert "bbox_occupancy_ratio" in obj.score_targets

    def test_from_json_text(self):
        json_text = '{"occupancy":0.4,"aspect_ratio":1.0,"center_x":0.5,"center_y":0.5}'
        obj = build_target_objective(target_json_text=json_text)
        assert obj.name == "custom_target"
        assert "bbox_occupancy_ratio" in obj.score_targets

    def test_preset_plus_json_override(self):
        obj = build_target_objective(
            preset_name="centered_50",
            target_json_text='{"bbox_occupancy_ratio":0.6}',
        )
        assert obj.score_targets["bbox_occupancy_ratio"] == 0.6

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown target preset"):
            build_target_objective(preset_name="nonexistent")

    def test_with_weight_json(self):
        obj = build_target_objective(
            preset_name="centered_50",
            score_weights_json_text='{"bbox_occupancy_ratio":5.0}',
        )
        assert obj.score_weights["bbox_occupancy_ratio"] == 5.0

    def test_returns_target_objective_dataclass(self):
        obj = build_target_objective(preset_name="centered_50")
        assert hasattr(obj, "name")
        assert hasattr(obj, "raw_spec")
        assert hasattr(obj, "score_targets")
        assert hasattr(obj, "score_weights")
