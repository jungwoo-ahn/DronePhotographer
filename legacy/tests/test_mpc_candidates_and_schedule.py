"""Tests for axis-only candidate mode and objective schedule."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.vlm_qwen25.mpc import generate_local_candidate_actions
from src.vlm_qwen25.objective import (
    build_target_objective,
    build_target_objective_schedule,
    objective_for_step,
)


def _basis():
    position = np.array([0.0, -5.0, 1.5], dtype=np.float32)
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return position, forward, up


# ---------------------------------------------------------------------------
# axis-only candidate generation
# ---------------------------------------------------------------------------


class TestAxisMode:
    def test_axis_mode_counts_and_single_axis_nonzero(self):
        position, forward, up = _basis()
        translation_values = [-0.2, -0.1, 0.1, 0.2]
        rotation_values_rad = [-0.05, 0.05]

        candidates = generate_local_candidate_actions(
            position=position,
            forward=forward,
            up=up,
            translation_values=translation_values,
            rotation_values_rad=rotation_values_rad,
            max_translation_norm=1.0,
            max_rotation_norm_rad=1.0,
            action_frame="camera_local",
            rotation_representation="orientation_6d",
            disable_roll=True,
            mode="axis",
        )

        # 3 axes * 4 nonzero translations + 2 axes (roll disabled) * 2 nonzero rotations + 1 no-op
        assert len(candidates) == 3 * 4 + 2 * 2 + 1

        for c in candidates:
            dp = np.asarray(c.delta_position_local, dtype=np.float32)
            dr = np.asarray(c.delta_rotation_local, dtype=np.float32)
            combined = np.concatenate([dp, dr])
            nonzero = int(np.sum(np.abs(combined) > 1e-8))
            assert nonzero <= 1, f"axis-only candidate must have at most one nonzero delta, got {combined}"

    def test_axis_mode_respects_max_norm(self):
        position, forward, up = _basis()
        translation_values = [-0.5, -0.1, 0.1, 0.5]

        candidates = generate_local_candidate_actions(
            position=position,
            forward=forward,
            up=up,
            translation_values=translation_values,
            rotation_values_rad=[],
            max_translation_norm=0.2,  # excludes ±0.5
            max_rotation_norm_rad=1.0,
            action_frame="camera_local",
            rotation_representation="orientation_6d",
            disable_roll=True,
            mode="axis",
        )

        # 3 axes * 2 nonzero translations that fit + 1 no-op
        assert len(candidates) == 3 * 2 + 1

    def test_axis_mode_roll_enabled(self):
        position, forward, up = _basis()
        candidates = generate_local_candidate_actions(
            position=position,
            forward=forward,
            up=up,
            translation_values=[],
            rotation_values_rad=[-0.05, 0.05],
            max_translation_norm=1.0,
            max_rotation_norm_rad=1.0,
            action_frame="camera_local",
            rotation_representation="orientation_6d",
            disable_roll=False,
            mode="axis",
        )
        # 3 axes * 2 nonzero rotations + 1 no-op
        assert len(candidates) == 3 * 2 + 1

    def test_grid_mode_unchanged(self):
        position, forward, up = _basis()
        candidates = generate_local_candidate_actions(
            position=position,
            forward=forward,
            up=up,
            translation_values=[-0.25, 0.0, 0.25],
            rotation_values_rad=[math.radians(-6), 0.0, math.radians(6)],
            max_translation_norm=0.5,
            max_rotation_norm_rad=math.radians(10),
            action_frame="camera_local",
            rotation_representation="orientation_6d",
            disable_roll=True,
        )
        # Grid mode: exact count is config-dependent but should be >> axis mode equivalent.
        # Minimally check it's substantially larger than the axis-only lower bound.
        assert len(candidates) > 50

    def test_invalid_mode_raises(self):
        position, forward, up = _basis()
        with pytest.raises(ValueError, match="mode"):
            generate_local_candidate_actions(
                position=position,
                forward=forward,
                up=up,
                translation_values=[0.1],
                rotation_values_rad=[0.05],
                max_translation_norm=1.0,
                max_rotation_norm_rad=1.0,
                action_frame="camera_local",
                rotation_representation="orientation_6d",
                mode="bogus",
            )


# ---------------------------------------------------------------------------
# objective schedule
# ---------------------------------------------------------------------------


class TestObjectiveSchedule:
    def test_two_phase_weights_switch(self):
        schedule_json = json.dumps([
            {
                "until_step": 30,
                "score_weights": {
                    "bbox_occupancy_ratio": 2.0,
                    "bbox_centroid_offset": 2.0,
                    "camera_to_object_fy": 0.0,
                },
            },
            {
                "until_step": 100,
                "score_weights": {
                    "bbox_occupancy_ratio": 0.5,
                    "bbox_centroid_offset": 0.5,
                    "camera_to_object_fy": 1.0,
                },
            },
        ])
        target_json = json.dumps({
            "bbox_centroid_offset": 0.0,
            "bbox_occupancy_ratio": 0.4,
            "camera_to_object_fy": 1.0,
        })
        schedule = build_target_objective_schedule(
            preset_name=None,
            target_json_text=target_json,
            schedule_json_text=schedule_json,
        )
        assert len(schedule) == 2

        early = objective_for_step(schedule, 0)
        late = objective_for_step(schedule, 50)

        assert early.score_weights["camera_to_object_fy"] == 0.0
        assert late.score_weights["camera_to_object_fy"] == 1.0
        assert early.score_weights["bbox_occupancy_ratio"] == 2.0
        assert late.score_weights["bbox_occupancy_ratio"] == 0.5

    def test_out_of_range_clamps_to_last(self):
        schedule_json = json.dumps([
            {"until_step": 5, "score_weights": {"bbox_occupancy_ratio": 1.0}},
            {"until_step": 10, "score_weights": {"bbox_occupancy_ratio": 3.0}},
        ])
        target_json = json.dumps({"bbox_occupancy_ratio": 0.4})
        schedule = build_target_objective_schedule(
            preset_name=None,
            target_json_text=target_json,
            schedule_json_text=schedule_json,
        )
        assert objective_for_step(schedule, 9).score_weights["bbox_occupancy_ratio"] == 3.0
        assert objective_for_step(schedule, 100).score_weights["bbox_occupancy_ratio"] == 3.0

    def test_schedule_targets_shared(self):
        """score_targets should be identical across phases; only weights change."""
        schedule_json = json.dumps([
            {"until_step": 5, "score_weights": {"bbox_occupancy_ratio": 1.0}},
            {"until_step": 10, "score_weights": {"bbox_occupancy_ratio": 2.0}},
        ])
        target_json = json.dumps({
            "bbox_centroid_offset": 0.0,
            "bbox_occupancy_ratio": 0.4,
        })
        schedule = build_target_objective_schedule(
            preset_name=None,
            target_json_text=target_json,
            schedule_json_text=schedule_json,
        )
        assert schedule[0][1].score_targets == schedule[1][1].score_targets

    def test_default_weights_fallback(self):
        """When a phase doesn't override a key, default_weights_json_text should supply it."""
        schedule_json = json.dumps([
            {"until_step": 5, "score_weights": {"bbox_occupancy_ratio": 2.0}},
        ])
        target_json = json.dumps({
            "bbox_centroid_offset": 0.0,
            "bbox_occupancy_ratio": 0.4,
        })
        default_weights = json.dumps({"bbox_centroid_offset": 3.0})
        schedule = build_target_objective_schedule(
            preset_name=None,
            target_json_text=target_json,
            schedule_json_text=schedule_json,
            default_weights_json_text=default_weights,
        )
        obj = objective_for_step(schedule, 0)
        assert obj.score_weights["bbox_centroid_offset"] == 3.0
        assert obj.score_weights["bbox_occupancy_ratio"] == 2.0

    def test_non_monotonic_until_step_raises(self):
        schedule_json = json.dumps([
            {"until_step": 10, "score_weights": {"bbox_occupancy_ratio": 1.0}},
            {"until_step": 5, "score_weights": {"bbox_occupancy_ratio": 2.0}},
        ])
        target_json = json.dumps({"bbox_occupancy_ratio": 0.4})
        with pytest.raises(ValueError, match="strictly greater"):
            build_target_objective_schedule(
                preset_name=None,
                target_json_text=target_json,
                schedule_json_text=schedule_json,
            )
