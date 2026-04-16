"""Tests for scripts/generate_target.py pure functions."""
from __future__ import annotations

import json
import math

import pytest

from generate_target import extract_json, validate_target, describe_target
from tests.conftest import dot3, norm3


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_empty_object(self):
        assert extract_json("{}") == {}

    def test_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_leading_text(self):
        text = 'Here is the JSON:\n{"a": 1}'
        assert extract_json(text) == {"a": 1}

    def test_trailing_text(self):
        text = '{"a": 1}\nDone.'
        assert extract_json(text) == {"a": 1}

    def test_leading_and_trailing_text(self):
        text = 'Sure!\n{"x": 0.5}\nHope that helps!'
        assert extract_json(text) == {"x": 0.5}

    def test_nested_braces(self):
        text = '{"outer": {"inner": 2}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": 2}}

    def test_fence_with_explanation(self):
        text = 'Sure!\n```json\n{"x":0.5}\n```\nHope that helps!'
        assert extract_json(text) == {"x": 0.5}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            extract_json("No json here")

    def test_only_open_brace_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            extract_json("{ incomplete")

    def test_multiple_json_objects_raises(self):
        # find("{") -> first, rfind("}") -> last => invalid combined span
        with pytest.raises(json.JSONDecodeError):
            extract_json('{"a":1} some text {"b":2}')

    def test_real_llm_output(self):
        """Realistic LLM output with full target."""
        text = (
            '```json\n'
            '{"center_x":0.5,"center_y":0.5,"occupancy":0.4,"aspect_ratio":1.0,'
            '"camera_to_object_fx":0.0,"camera_to_object_fy":1.0,"camera_to_object_fz":0.0,'
            '"camera_to_object_ux":0.0,"camera_to_object_uy":0.0,"camera_to_object_uz":1.0}\n'
            '```'
        )
        result = extract_json(text)
        assert result["occupancy"] == 0.4
        assert result["camera_to_object_fy"] == 1.0
        assert len(result) == 10


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------

class TestValidateTarget:
    def test_valid_orthogonal_unchanged(self, minimal_valid_target):
        result = validate_target(minimal_valid_target)
        assert result["camera_to_object_fy"] == 1.0
        assert result["camera_to_object_uz"] == 1.0

    def test_missing_c2o_key_raises(self):
        target = {
            "occupancy": 0.4,
            "camera_to_object_fx": 0.0,
            "camera_to_object_fy": 1.0,
            # missing fz
            "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 0.0,
            "camera_to_object_uz": 1.0,
        }
        with pytest.raises(ValueError, match="Missing required key"):
            validate_target(target)

    def test_missing_occupancy_raises(self):
        target = {
            "camera_to_object_fx": 0.0,
            "camera_to_object_fy": 1.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 0.0,
            "camera_to_object_uz": 1.0,
        }
        with pytest.raises(ValueError, match="Missing required key: occupancy"):
            validate_target(target)

    def test_non_orthogonal_auto_correction(self):
        target = {
            "occupancy": 0.4,
            "camera_to_object_fx": 0.5,
            "camera_to_object_fy": 0.5,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.5,
            "camera_to_object_uy": 0.5,
            "camera_to_object_uz": 0.5,
        }
        # dot(f, u) = 0.25 + 0.25 + 0 = 0.5 > 0.1, should auto-correct
        result = validate_target(target)
        f = (result["camera_to_object_fx"], result["camera_to_object_fy"], result["camera_to_object_fz"])
        u = (result["camera_to_object_ux"], result["camera_to_object_uy"], result["camera_to_object_uz"])
        assert abs(dot3(f, u)) < 0.02
        assert abs(norm3(u) - 1.0) < 0.02
        # f vector should remain unchanged
        assert result["camera_to_object_fx"] == 0.5
        assert result["camera_to_object_fy"] == 0.5

    def test_slightly_non_orthogonal_no_change(self):
        # dot = 0.08 < 0.1 threshold
        target = {
            "occupancy": 0.4,
            "camera_to_object_fx": 1.0,
            "camera_to_object_fy": 0.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.08,
            "camera_to_object_uy": 0.0,
            "camera_to_object_uz": 1.0,
        }
        result = validate_target(target)
        assert result["camera_to_object_ux"] == 0.08  # unchanged

    def test_already_orthogonal_unit(self):
        target = {
            "occupancy": 0.3,
            "camera_to_object_fx": 1.0,
            "camera_to_object_fy": 0.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 1.0,
            "camera_to_object_uz": 0.0,
        }
        result = validate_target(target)
        assert result["camera_to_object_fx"] == 1.0
        assert result["camera_to_object_uy"] == 1.0

    def test_preserves_extra_keys(self, valid_full_target):
        result = validate_target(valid_full_target)
        assert result["center_x"] == 0.5
        assert result["center_y"] == 0.5
        assert result["aspect_ratio"] == 1.0

    def test_zero_f_norm_no_crash(self):
        target = {
            "occupancy": 0.4,
            "camera_to_object_fx": 0.0,
            "camera_to_object_fy": 0.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.5,
            "camera_to_object_uy": 0.5,
            "camera_to_object_uz": 0.0,
        }
        # dot = 0, so orthogonalization won't trigger (|dot| = 0 < 0.1)
        result = validate_target(target)
        assert result is not None


# ---------------------------------------------------------------------------
# describe_target (thin wrapper around _describe_targets)
# ---------------------------------------------------------------------------

class TestDescribeTarget:
    def test_front_eyelevel(self):
        target = {
            "occupancy": 0.4,
            "camera_to_object_fx": 0.0,
            "camera_to_object_fy": 1.0,
            "camera_to_object_fz": 0.0,
            "camera_to_object_ux": 0.0,
            "camera_to_object_uy": 0.0,
            "camera_to_object_uz": 1.0,
            "bbox_occupancy_ratio": 0.4,
            "bbox_centroid_offset": 0.0,
        }
        desc = describe_target(target)
        assert "front" in desc
        assert "eye-level" in desc

    def test_returns_string(self, minimal_valid_target):
        # describe_target builds weights from target keys; c2o keys
        # need to be present for _describe_targets to work
        desc = describe_target(minimal_valid_target)
        assert isinstance(desc, str)
