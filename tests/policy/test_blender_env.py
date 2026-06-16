"""BlenderRolloutEnv logic tests using MockRenderer (no Blender binary needed)."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from src.policy.common.blender_env import BlenderRolloutEnv, MockRenderer, pose_proxy_distance

_POS = [0.0, -5.0, 1.0]
_FWD = [0.0, 1.0, 0.0]
_UP = [0.0, 0.0, 1.0]


def _env(**kw):
    return BlenderRolloutEnv("dummy_run_info.json", MockRenderer(), **kw)


def test_reset_renders_and_returns_obs():
    env = _env()
    obs = env.reset(_POS, _FWD, _UP)
    assert obs["image"] is not None
    assert obs["t"] == 0
    np.testing.assert_allclose(obs["pose"]["position"], _POS)
    assert len(env.renderer.calls) == 1


def test_reset_no_render():
    env = _env()
    obs = env.reset(_POS, _FWD, _UP, render=False)
    assert obs["image"] is None
    assert len(env.renderer.calls) == 0


def test_step_zero_action_keeps_pose():
    env = _env()
    env.reset(_POS, _FWD, _UP)
    obs, info = env.step(np.zeros(5, dtype=np.float32))
    np.testing.assert_allclose(obs["pose"]["position"], _POS, atol=1e-5)
    assert info["t"] == 1 and obs["t"] == 1


def test_step_translation_moves_camera_and_rerenders():
    env = _env()
    env.reset(_POS, _FWD, _UP)
    before = env.position.copy()
    obs, _ = env.step(np.array([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32))  # dolly forward 2m
    assert not np.allclose(obs["pose"]["position"], before)
    assert len(env.renderer.calls) == 2          # reset + step
    # different pose -> different mock pixels
    import numpy as _np
    assert _np.asarray(env.renderer.render("x", before, _FWD, _UP)).mean() != \
        _np.asarray(env.renderer.render("x", env.position, _FWD, _UP)).mean()


def test_step_before_reset_raises():
    env = _env()
    with pytest.raises(RuntimeError):
        env.step(np.zeros(5))


def test_pose_proxy_distance_az_el():
    target = {"cam_to_obj_azimuth_deg": 90.0, "cam_to_obj_elevation_deg": 0.0}
    keys = list(target)
    # object straight along +y from origin -> az = atan2(1,0) = 90 deg, el = 0 -> distance ~0
    d = pose_proxy_distance(np.array([0.0, 0.0, 0.0]), np.array([0.0, 5.0, 0.0]), target, keys)
    assert d is not None and d < 1e-3


def test_pose_proxy_distance_none_without_az_el_keys():
    env = _env(object_position=[0.0, 5.0, 0.0])
    env.reset(_POS, _FWD, _UP, render=False)
    assert env.pose_proxy_distance({"occupancy": 35.0}, ["occupancy"]) is None


def test_env_pose_proxy_uses_current_pose():
    env = _env(object_position=[0.0, 5.0, 0.0])
    env.reset(_POS, _FWD, _UP, render=False)
    target = {"cam_to_obj_azimuth_deg": 90.0, "cam_to_obj_elevation_deg": 0.0}
    d = env.pose_proxy_distance(target, list(target))
    assert isinstance(d, float)
