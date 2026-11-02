"""Render profiles, the v7 Blender wrapper's dispatch, and the GIF pose interpolation.

The v7 profile exists because the rollout renderer drew a different picture from the training
renderer for the same pose (MAE 24-101 /255 per held-out scene). Three things have to hold for it
to do anything at all, and each fails silently if broken: the options must reach the run_info,
the renderer must launch the V12 wrapper instead of the shared script (BlenderDrone ignores
`v7_renderer` on its own), and the wrapper must hand each argument list to the right shared
script.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))

from src.common.run_info import RENDER_PROFILES, renderer_kwargs, write_run_info  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, V12 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy_profile_changes_nothing():
    assert RENDER_PROFILES["legacy"] == {}
    assert renderer_kwargs("legacy", persistent=True) == {}
    assert renderer_kwargs("legacy", persistent=False) == {}


def test_v7_profile_matches_the_training_renderer():
    # scripts/v7_reference/v7_sample_pairs_smoke.configure_renderer
    v7 = RENDER_PROFILES["v7"]
    assert v7["samples"] == 32 and v7["sky_strength"] == 0.1 and v7["v7_renderer"] is True


@pytest.mark.parametrize("persistent,key", [(True, "server_script"), (False, "render_script")])
def test_v7_profile_launches_the_wrapper(persistent, key):
    kw = renderer_kwargs("v7", persistent=persistent)
    assert list(kw) == [key]
    assert Path(kw[key]).is_absolute() and Path(kw[key]).exists()
    assert Path(kw[key]).name == "blender_render_v7.py"


def test_v7_options_reach_the_run_info(tmp_path):
    root = V12 / "data" / "trajectories" / "v7_stage2_renders_lookat075"
    placement = ("Desert-Roadside-Repair-Garage_a1a2b2ac__young-guy-keeps-_"
                 "ad9f69fc-9bc7-461f-95c3-1026c06b6797")
    data = root / placement / "data.json"
    if not data.exists():
        pytest.skip("v7 data not present")
    ri = write_run_info(placement, data, tmp_path, shared_root="/home/nas_main/jungwooahn/projects/"
                        "DronePhotographer", resolution=(1024, 768),
                        extra_options={"render_device": "GPU", **RENDER_PROFILES["v7"]})
    opts = json.loads(Path(ri).read_text())["options"]
    assert opts["sky_strength"] == 0.1 and opts["samples"] == 32 and opts["v7_renderer"] is True
    assert opts["resolution"] == [1024, 768]


def test_wrapper_dispatches_by_arguments():
    w = _load("blender_render_v7")
    assert w.target_script(["blender", "-b", "-P", "x.py", "--", "--run_info_path", "r",
                            "--comm_dir", "/tmp/c"]) == w.SERVER
    assert w.target_script(["blender", "-b", "-P", "x.py", "--", "--run_info_path", "r",
                            "--output_image", "o.png", "--output_json", "o.json"]) == w.POSE
    with pytest.raises(SystemExit):
        w.target_script(["blender", "--", "--run_info_path", "r"])


def test_interpolation_keeps_endpoints_and_an_orthonormal_frame():
    f = _load("render_trajectory_frames")
    p0 = {"position": [0.0, 0.0, 1.0], "forward": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}
    a = np.radians(4.0)
    p1 = {"position": [0.2, 0.1, 1.0], "forward": [np.cos(a), np.sin(a), 0.0], "up": [0.0, 0.0, 1.0]}
    poses = f.frame_poses([p0, p1, p0], sub=2)
    assert len(poses) == 1 + 2 * 3                        # start, then (2 interpolated + step) x 2
    assert poses[3][0] is p1 and poses[3][1] == 1 and poses[3][2] == 0.0
    mid = f.interpolate(p0, p1, 0.5)
    fw, up = np.asarray(mid["forward"]), np.asarray(mid["up"])
    assert np.isclose(np.linalg.norm(fw), 1) and np.isclose(np.linalg.norm(up), 1)
    assert abs(float(np.dot(fw, up))) < 1e-9
    assert np.allclose(mid["position"], [0.1, 0.05, 1.0])
