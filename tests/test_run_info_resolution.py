"""`write_run_info` — the placement transform, and the two render knobs added for goal rendering.

The transform itself is what this adapter exists to get right (a second copy of it is a second
chance to render a scene whose subject sits somewhere else); these pin the parts that changed:
a non-square resolution, and the render options a full-resolution render needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH
from src.common.run_info import render_resolution, to_policy_frame, write_run_info

SHARED = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")
ROOT = Path("data/trajectories/v7_stage2_renders_lookat075")
_needs = pytest.mark.skipif(not ROOT.is_dir(), reason="no trajectory data in this checkout")


def _any_placement():
    for d in sorted(ROOT.iterdir()):
        if (d / "data.json").exists():
            return d.name, d / "data.json"
    pytest.skip("no placement with data.json")


@_needs
def test_an_int_renders_at_the_training_aspect_not_a_square(tmp_path):
    """An int is the POLICY's square input size, not the render shape.

    Rendering the square natively keeps the horizontal FOV and widens the vertical one by
    4/3 (Blender's AUTO sensor fit keys off the long side), so the subject comes out 0.75x
    as tall as in training, where frames were rendered 1024x768 and squashed by ffmpeg.
    """
    name, data = _any_placement()
    ri = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED)).read_text())
    w, h = ri["options"]["resolution"]
    assert w != h, "a native square render is the FOV bug this replaced"
    assert w * int(RENDER_HEIGHT) == h * int(RENDER_WIDTH), (w, h)
    assert h >= 256, "the squash to the policy square must only ever downsample"


def test_render_resolution_is_exactly_the_training_aspect():
    for n in (224, 256, 384):
        w, h = render_resolution(n)
        assert w * int(RENDER_HEIGHT) == h * int(RENDER_WIDTH), (n, w, h)
        assert h >= n


def test_to_policy_frame_squashes_rather_than_crops():
    """The training squash is non-aspect-preserving; a crop or a letterbox is a different
    picture. Checked on content, not just shape: a marker at the frame's right edge has to
    survive at the right edge."""
    import numpy as np

    img = np.zeros((258, 344, 3), dtype=np.uint8)
    img[:, -8:, 0] = 255                      # right-edge stripe
    out = to_policy_frame(img, 256)
    assert out.shape == (256, 256, 3)
    assert out[:, -1, 0].mean() > 200, "right edge lost -> cropped, not squashed"
    assert out[:, 0, 0].mean() < 50


def test_to_policy_frame_passes_a_square_through():
    import numpy as np

    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    assert to_policy_frame(img, 256) is img


@_needs
def test_a_pair_is_kept_as_width_height(tmp_path):
    """The goal profile's pixel keys are defined on the 1024x768 frame; a square render would
    change the field of view and make the same goal a different shot."""
    name, data = _any_placement()
    ri = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED,
                                        resolution=(1024, 768))).read_text())
    assert ri["options"]["resolution"] == [1024, 768]


@_needs
def test_extra_options_merge_without_displacing_the_transform(tmp_path):
    name, data = _any_placement()
    p = write_run_info(name, data, tmp_path, shared_root=SHARED, resolution=(1024, 768),
                       extra_options={"render_device": "GPU", "samples": 32})
    ri = json.loads(Path(p).read_text())
    assert ri["options"]["render_device"] == "GPU" and ri["options"]["samples"] == 32
    assert ri["options"]["resolution"] == [1024, 768]
    # the transform keys BlenderDrone reconstructs the placement from must all survive
    for k in ("input_scene", "input_object", "scene_scale", "scale", "rotation_xyz_rad"):
        assert k in ri, k
    assert len(ri["options"]["object_position"]) == 3


@_needs
def test_extra_options_cannot_silently_override_the_object_position(tmp_path):
    """A caller passing `object_position` would move the subject; the merge order puts
    extra_options last, so this test states what that costs and keeps it visible."""
    name, data = _any_placement()
    base = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED)).read_text())
    p = write_run_info(name, data, tmp_path, shared_root=SHARED,
                       extra_options={"samples": 8})
    assert json.loads(Path(p).read_text())["options"]["object_position"] == base["options"]["object_position"]
