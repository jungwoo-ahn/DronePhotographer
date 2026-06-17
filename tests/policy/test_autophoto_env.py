"""PhotoEnv tests with MockRenderer + a mock reward (no Blender, no real scorer)."""

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("gymnasium")

from src.policy.autophoto.env import PHOTO_ACTIONS, PhotoEnv
from src.policy.common.blender_env import BlenderRolloutEnv, MockRenderer
from src.policy.common.validation_sample import ValidationSample


class _MockReward:
    feature_dim = 512

    def score_and_features(self, image):
        m = float(np.asarray(image).mean())
        return m, np.full(512, m, dtype=np.float32)


def _sample():
    return ValidationSample(
        placement="S__O", scene_file="s.blend", object_file="o.blend", scene_scale=1.0,
        object_position=[0, 0, 0], object_rotation_xyz=[0, 0, 0], object_scale=1.0,
        subject_center=[0, 5, 0], render_width=64, render_height=64, render_samples=1,
        start_poses=[{"pos": [0, -5, 1], "forward": [0, 1, 0], "up": [0, 0, 1]}],
    )


def _env(**kw):
    rollout = BlenderRolloutEnv.from_validation_sample(_sample(), MockRenderer())
    return PhotoEnv(rollout, _MockReward(), **kw)


def test_action_map_is_9_with_terminate():
    assert len(PHOTO_ACTIONS) == 9
    assert PHOTO_ACTIONS[3].terminate
    assert PHOTO_ACTIONS[0].delta[2] > 0     # forward -> +dz
    assert PHOTO_ACTIONS[4].delta[2] < 0     # back -> -dz
    assert PHOTO_ACTIONS[1].delta[3] < 0     # turn_left -> -yaw
    assert PHOTO_ACTIONS[2].delta[3] > 0     # turn_right -> +yaw


def test_spaces():
    env = _env()
    assert env.action_space.n == 9
    assert env.observation_space.shape == (512,)
    env.close()


def test_reset_returns_features_and_score():
    env = _env()
    obs, info = env.reset()
    assert obs.shape == (512,)
    assert "init_score" in info
    env.close()


def test_step_moves_camera_and_shapes_reward():
    env = _env()
    env.reset()
    before = env.rollout.position.copy()
    obs, reward, terminated, truncated, info = env.step(0)   # forward
    assert obs.shape == (512,)
    assert not terminated
    assert not np.allclose(env.rollout.position, before)     # camera dollied
    assert isinstance(reward, float)
    env.close()


def test_terminate_action_ends_episode():
    env = _env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)   # terminate
    assert terminated and not truncated
    assert "final_score" in info
    env.close()


def test_truncates_at_max_steps():
    env = _env(max_steps=3)
    env.reset()
    done = trunc = False
    n = 0
    while not (done or trunc) and n < 10:
        _, _, done, trunc, _ = env.step(0)   # keep going forward
        n += 1
    assert trunc and n == 3
    env.close()


def test_sb3_check_env_passes():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    env = _env()
    check_env(env, warn=True, skip_render_check=True)
    env.close()
