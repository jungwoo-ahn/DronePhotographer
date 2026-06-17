"""Tests for the v7 validation-sample resolver (data.json + VLM v6 record join)."""

import json
import math

import pytest

from src.policy.common.validation_sample import load_validation_sample

_PLACEMENT = "Scene-A_xxx__Object-B_yyy"


def _write_sample(tmp_path, *, scene_scale=0.8333, accepted_rotation=(0.1, 0.0, 0.2),
                  accepted_scale=1.0, with_vlm=True):
    data = {
        "placement": _PLACEMENT,
        "scene_file": "data/scenes/Scene-A_xxx/Scene-A_xxx.blend",
        "object_file": "data/objects/Object-B_yyy/Object-B_yyy.blend",
        "subject_center": [1.0, 2.0, 0.9],
        "subject_foot": [1.0, 2.0, 0.0],
        "render_width": 1024, "render_height": 768, "render_samples": 32,
        "accepted_pairs": [
            {"start": {"pos": [0, -5, 1], "forward": [0, 1, 0], "up": [0, 0, 1]}},
            {"start": {"pos": [3, -4, 1], "forward": [-0.6, 0.8, 0], "up": [0, 0, 1]}},
        ],
    }
    data_json = tmp_path / "data.json"
    data_json.write_text(json.dumps(data))
    vlm_dir = tmp_path / "vlm_v6"
    vlm_dir.mkdir()
    if with_vlm:
        vlm = {
            "scene": "Scene-A_xxx", "scene_file": data["scene_file"],
            "object_file": data["object_file"], "scene_scale": scene_scale,
            "placements": [
                {"accepted": False, "position": [9, 9, 9], "rotation": [0, 0, 0], "scale": 5.0},
                {"accepted": True, "position": [1.0, 2.0, 0.0],
                 "rotation": list(accepted_rotation), "scale": accepted_scale},
            ],
        }
        (vlm_dir / f"{_PLACEMENT}.json").write_text(json.dumps(vlm))
    return data_json, vlm_dir


def test_resolves_transform_from_vlm_record(tmp_path):
    data_json, vlm_dir = _write_sample(tmp_path, scene_scale=0.75, accepted_rotation=(0.1, 0.2, 0.3))
    s = load_validation_sample(data_json, vlm_dir)
    assert s.placement == _PLACEMENT
    assert s.scene_scale == pytest.approx(0.75)
    assert s.object_rotation_xyz == pytest.approx([0.1, 0.2, 0.3])   # from the ACCEPTED candidate
    assert s.object_position == pytest.approx([1.0, 2.0, 0.0])
    assert s.subject_center == pytest.approx([1.0, 2.0, 0.9])
    assert s.render_width == 1024 and s.render_samples == 32
    assert len(s.start_poses) == 2


def test_to_run_info_carries_scene_scale_and_full_rotation(tmp_path):
    data_json, vlm_dir = _write_sample(tmp_path, scene_scale=0.9, accepted_rotation=(0.0, 0.0, 1.57))
    ri = load_validation_sample(data_json, vlm_dir).to_run_info()
    assert ri["scene_scale"] == pytest.approx(0.9)
    assert ri["rotation_xyz_rad"] == pytest.approx([0.0, 0.0, 1.57])
    assert ri["input_scene"].endswith(".blend")
    assert ri["options"]["object_position"] == pytest.approx([1.0, 2.0, 0.0])
    assert ri["options"]["resolution"] == [1024, 768]
    assert ri["options"]["focal_length"] == pytest.approx(24.0)


def test_fov_from_intrinsics(tmp_path):
    data_json, vlm_dir = _write_sample(tmp_path)
    s = load_validation_sample(data_json, vlm_dir)
    assert s.hfov_deg() == pytest.approx(math.degrees(2 * math.atan(12.8 / 48.0)), abs=1e-6)  # ~29.86
    assert s.vfov_deg() == pytest.approx(math.degrees(2 * math.atan(9.6 / 48.0)), abs=1e-6)   # ~22.62


def test_start_pose_accessor(tmp_path):
    data_json, vlm_dir = _write_sample(tmp_path)
    s = load_validation_sample(data_json, vlm_dir)
    pos, fwd, up = s.start_pose(1)
    assert pos == [3, -4, 1] and up == [0, 0, 1]


def test_missing_vlm_record_raises_unless_fallback(tmp_path):
    data_json, vlm_dir = _write_sample(tmp_path, with_vlm=False)
    with pytest.raises(FileNotFoundError):
        load_validation_sample(data_json, vlm_dir)
    s = load_validation_sample(data_json, vlm_dir, require_vlm=False)
    assert s.scene_scale == 1.0 and s.object_scale == 1.0
    assert s.object_rotation_xyz == [0.0, 0.0, 0.0]
    assert s.object_position == pytest.approx([1.0, 2.0, 0.0])   # falls back to subject_foot
