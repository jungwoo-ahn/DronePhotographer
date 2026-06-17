"""BlenderRolloutEnv.from_validation_sample / reset_to_start (MockRenderer, no Blender)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from src.policy.common.blender_env import BlenderRolloutEnv, MockRenderer
from src.policy.common.validation_sample import ValidationSample


def _sample():
    return ValidationSample(
        placement="Scene__Obj",
        scene_file="data/scenes/S/S.blend",
        object_file="data/objects/O/O.blend",
        scene_scale=0.8,
        object_position=[1.0, 2.0, 0.0],
        object_rotation_xyz=[0.0, 0.0, 0.5],
        object_scale=1.0,
        subject_center=[1.0, 2.0, 0.9],
        render_width=1024, render_height=768, render_samples=32,
        start_poses=[{"pos": [0, -5, 1], "forward": [0, 1, 0], "up": [0, 0, 1]}],
    )


def test_from_validation_sample_writes_run_info_and_sets_object():
    env = BlenderRolloutEnv.from_validation_sample(_sample(), MockRenderer())
    ri = json.loads(Path(env.run_info_path).read_text())
    assert ri["scene_scale"] == pytest.approx(0.8)
    assert ri["rotation_xyz_rad"] == pytest.approx([0.0, 0.0, 0.5])
    np.testing.assert_allclose(env.object_position, [1.0, 2.0, 0.9])   # subject center
    env.close()


def test_reset_to_start_uses_recorded_pose_and_renders():
    env = BlenderRolloutEnv.from_validation_sample(_sample(), MockRenderer())
    obs = env.reset_to_start(0)
    np.testing.assert_allclose(obs["pose"]["position"], [0, -5, 1])
    assert obs["image"] is not None
    assert len(env.renderer.calls) == 1
    env.close()


def test_close_removes_temp_run_info():
    env = BlenderRolloutEnv.from_validation_sample(_sample(), MockRenderer())
    p = env.run_info_path
    assert Path(p).exists()
    env.close()
    assert not Path(p).exists()      # temp run_info cleaned up


def test_pose_proxy_after_reset_to_start():
    env = BlenderRolloutEnv.from_validation_sample(_sample(), MockRenderer())
    env.reset_to_start(0, render=False)
    d = env.pose_proxy_distance({"cam_to_obj_azimuth_deg": 90.0, "cam_to_obj_elevation_deg": 0.0},
                                ["cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"])
    assert isinstance(d, float)
    env.close()
