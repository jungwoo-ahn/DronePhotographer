"""Plumbing tests for the closed-loop rollout eval (MockRenderer + mock policy, CPU).

Validates the driver loop, the geometric scorer, and goal selection without a
Blender binary or a real checkpoint. The real model + Blender are exercised by the
login smoke / single real rollout (see scripts/rollout_eval.py docstring).
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.policy.common.blender_env import BlenderRolloutEnv, MockRenderer
from src.policy.common.reward import CameraIntrinsics
from src.scoring.bbox_control import V5_SCORE_KEYS
from src.scoring.projection import cam_to_subject_angles, score_pose

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("rollout_eval")
B = _load("build_eval_goals")


class _FakeSample:
    placement = "TestScene_abc__obj_def"
    subject_center = [0.0, 0.0, 1.0]
    render_width, render_height = 1024, 768
    focal_length, sensor_width, sensor_height = 24.0, 12.8, 9.6

    def to_run_info(self):
        return {"input_scene": "none"}

    def start_pose(self, i=0):
        return ([0.0, -5.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])


class _MockPolicy:
    def __init__(self, chunk=8):
        self.chunk = chunk

    def sample(self, image_latent, goal_vec, n_steps=32):
        chunk = torch.zeros(1, self.chunk, 5)
        chunk[:, :, 2] = 0.2  # small forward dolly each step
        return SimpleNamespace(pred_action_chunk=chunk, pred_value=None, pred_latents=None)


class _MockVAE:
    def encode_pair_frames(self, a, b):
        return torch.zeros(1, 16, 2, 4, 4)


def _cube(center):
    c = np.asarray(center, dtype=np.float64)
    offs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=np.float64)
    return c + offs


def test_score_pose_returns_full_int_profile():
    c = [0.0, 0.0, 1.0]
    prof = score_pose([0, -5, 1], [0, 1, 0], [0, 0, 1], _cube(c), c, 1024, 768, [-0.2667, 0.2667, -0.2, 0.2])
    assert set(prof) == set(V5_SCORE_KEYS)
    assert all(isinstance(v, int) for v in prof.values())
    assert 0 <= prof["occupancy"] <= 100


def test_cam_to_subject_angles_signs():
    # camera below the subject -> elevation positive (cam looks up); above -> negative
    _, el_below = cam_to_subject_angles([0, -5, 0.0], [0, 0, 5.0])
    _, el_above = cam_to_subject_angles([0, -5, 5.0], [0, 0, 0.0])
    assert el_below > 0 > el_above


def test_farthest_point_distinct():
    X = np.random.RandomState(0).randn(50, 8).astype(np.float32)
    idx = B.farthest_point(X, 10, 0)
    assert len(set(idx)) == 10


def test_success_and_distance_helpers():
    keys = list(V5_SCORE_KEYS)
    goal = {k: 0 for k in keys}
    goal.update(occupancy=30, body_in_frame_ratio=100, object_center_x=512,
                object_center_y=384, bbox_x_offset=100, bbox_y_offset=150)
    assert R.is_success(dict(goal), goal)                 # identical -> reached
    far = dict(goal); far["occupancy"] = 90
    assert not R.is_success(far, goal)
    assert R.dist_raw(far, goal, keys) > R.dist_raw(dict(goal), goal, keys)


def test_rollout_loop_mock():
    env = BlenderRolloutEnv.from_validation_sample(_FakeSample(), MockRenderer())
    geom = R.mock_geom([0.0, 0.0, 1.0])
    keys = list(V5_SCORE_KEYS)
    goal = {k: 0 for k in keys}
    goal.update(occupancy=30, body_in_frame_ratio=100, object_center_x=512,
                object_center_y=384, bbox_x_offset=100, bbox_y_offset=150)
    intr = CameraIntrinsics.from_render(1024, 768)
    try:
        summ, frames = R.rollout(
            env, _MockPolicy(8), _MockVAE(), geom, goal, keys, [0.0, 0.0, 1.0],
            max_steps=3, execute_k=1, n_steps=2,
            device=torch.device("cpu"), dtype=torch.float32, intr=intr)
    finally:
        env.close()
    assert summ["n_steps"] <= 3
    assert len(frames) == len(summ["steps"])
    assert {"d_start", "d_final", "improvement", "reached"} <= set(summ)
    assert all(set(s["profile"]) == set(V5_SCORE_KEYS) for s in summ["steps"])
