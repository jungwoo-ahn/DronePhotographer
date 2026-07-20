"""Bidirectional multi-scale endpoint sampling — the fix for actions ignoring the goal.

Per start frame we emit one window per signed offset ±8/±16/±24 whose endpoint exists;
the goal is that endpoint, so the SAME start with DIFFERENT endpoints has DIFFERENT action
targets. These tests confirm the window count/indexing, the strided re-encode ("merge"),
the per-step value sequence, and — the point of the whole scheme — that actions vary with
the goal at the data level.
"""
import itertools
from collections import Counter, defaultdict

import numpy as np
import pytest

from src.policy.common.action_repr import encode_action_5d
from src.policy.common.annotations import iter_multiscale_windows
from src.policy.common.dataset_base import BasePolicyDataset, _compute_action_chunk, resolve_value_spec
from src.policy.common.goal_space import goal_keys, goal_vector, normalize_goal
from src.policy.common.reward import VALUE_SCALE, pose_distance_value

# reuse the synthetic v7 placement builder from the integration test
from tests.policy.test_v7_integration import _build_dummy_v7_placement


def _norm_profile(view, keys):
    return normalize_goal(np.nan_to_num(goal_vector(view.raw, keys)), keys)


@pytest.fixture
def placement(tmp_path):
    return _build_dummy_v7_placement(tmp_path / "Scene_aaaa__Obj_bbbb",
                                     n_accepted_pairs=1, frames_per_pair=32)


def test_window_count_is_96_per_pair(placement):
    w = list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8, 16, 24)))
    assert len(w) == 96                    # 24+16+8 forward + 24+16+8 reverse


def test_offset_direction_breakdown(placement):
    w = list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8, 16, 24)))
    c = Counter((abs(x.end_frame_idx - x.start_frame_idx), x.direction) for x in w)
    assert c[(8, 1)] == 24 and c[(8, -1)] == 24
    assert c[(16, 1)] == 16 and c[(16, -1)] == 16
    assert c[(24, 1)] == 8 and c[(24, -1)] == 8


def test_strided_reencode_is_the_merge(placement):
    """offset-16 → frame_step 2: action[k] is re-encoded between strided keyframes
    kf[k]→kf[k+1] (== frames 2k apart), NOT a sum of single-step deltas."""
    w = list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8, 16, 24)))
    w16 = next(x for x in w if x.end_frame_idx - x.start_frame_idx == 16 and x.direction == 1)
    assert w16.frame_step == 2 and len(w16.keyframes) == 9
    # keyframes are strided by 2 across the 16-frame span
    assert [f.frame_idx for f in w16.keyframes] == list(range(w16.start_frame_idx, w16.start_frame_idx + 17, 2))
    ac = _compute_action_chunk(w16)
    assert ac.shape == (8, 5)
    for k in range(8):
        a, b = w16.keyframes[k], w16.keyframes[k + 1]
        exp = encode_action_5d(
            np.asarray(a.camera_position, np.float32), np.asarray(a.camera_forward, np.float32), np.asarray(a.camera_up, np.float32),
            np.asarray(b.camera_position, np.float32), np.asarray(b.camera_forward, np.float32), np.asarray(b.camera_up, np.float32),
        )
        np.testing.assert_allclose(ac[k], exp, atol=1e-5)


def test_negative_offset_plays_trajectory_backward(placement):
    """Negative offsets play the path backward: keyframes descend, and the forward
    p→p+8 / reverse (p+8)→p windows cover the same unordered frame pairs. (The dolly
    direction flip is a data property tested on real trajectories in test_trajectory_reverse.)"""
    w = list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8,)))
    fwd = [x for x in w if x.direction == 1]
    rev = [x for x in w if x.direction == -1]
    assert fwd and rev
    for x in rev:
        idxs = [f.frame_idx for f in x.keyframes]
        assert idxs == sorted(idxs, reverse=True) and x.start_frame_idx > x.end_frame_idx
    for x in fwd:
        idxs = [f.frame_idx for f in x.keyframes]
        assert idxs == sorted(idxs) and x.start_frame_idx < x.end_frame_idx
    assert ({(x.start_frame_idx, x.end_frame_idx) for x in fwd}
            == {(x.end_frame_idx, x.start_frame_idx) for x in rev})


def test_offset_must_be_multiple_of_chunk_size(placement):
    with pytest.raises(ValueError):
        list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8, 12)))   # 12 % 8 != 0


def test_dataset_value_sequence_and_goal_pinned(placement):
    ds = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir", offsets=(8, 16, 24))
    assert len(ds) > 0
    s = ds[0]
    assert s.action_chunk.shape == (8, 5)
    assert s.value.shape == (8,)
    assert s.goal.frame_idx == s.end.frame_idx            # goal pinned to the endpoint
    # value[0] is the START→GOAL scalar; the sequence is the per-step cost-to-go
    v0 = pose_distance_value(
        s.start.camera_position, s.start.camera_forward, s.start.camera_up,
        s.goal.camera_position, s.goal.camera_forward, s.goal.camera_up,
        subject_center=s.start.subject_center, subject_height=s.start.subject_height,
    )
    assert s.value[0] == pytest.approx(v0, abs=1e-5)
    assert np.isfinite(s.value).all()


def test_actions_depend_on_goal_at_data_level(placement):
    """The whole point: for a FIXED start, different endpoints ⇒ different action chunks."""
    w = list(iter_multiscale_windows(placement, chunk_size=8, offsets=(8, 16, 24)))
    by_start = defaultdict(list)
    for x in w:
        by_start[(x.pair_idx, x.start_frame_idx)].append(x)
    grp = max(by_start.values(), key=len)                  # the start with the most offsets
    assert len(grp) >= 2
    ends = [x.end_frame_idx for x in grp]
    assert len(set(ends)) == len(ends)                     # each window has a distinct goal
    chunks = [_compute_action_chunk(x) for x in grp]
    maxdiff = max(np.abs(a - b).max() for a, b in itertools.combinations(chunks, 2))
    assert maxdiff > 1e-3, "actions must differ across goals from the same start"


def test_multiscale_ignores_augment_reverse(placement):
    """Bidirectional offsets already include reverse; augment_reverse must not double-count."""
    a = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir",
                          offsets=(8, 16, 24), augment_reverse=False)
    b = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir",
                          offsets=(8, 16, 24), augment_reverse=True)
    assert len(a) == len(b)


# --- value_target_mode (cost_to_go | achieved_profile | profile_delta) --------------


def test_resolve_value_spec():
    assert resolve_value_spec("cost_to_go", 8) == (1, float(VALUE_SCALE))
    assert resolve_value_spec("achieved_profile", 8) == (8, 1.0)
    assert resolve_value_spec("profile_delta", 5) == (5, 1.0)
    with pytest.raises(ValueError):
        resolve_value_spec("bogus", 8)


def test_value_mode_default_is_cost_to_go(placement):
    ds = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir")
    assert ds[0].value.shape == (8,)   # backward-compatible default (scalar/step)


def test_value_mode_achieved_profile(placement):
    keys = goal_keys()
    ds = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir",
                           value_target_mode="achieved_profile")
    s = ds[0]
    assert s.value.shape == (8, len(keys))
    # value[k] = normalize_goal(profile(keyframe[k])); value[0] uses the start frame
    np.testing.assert_allclose(s.value[0], _norm_profile(s.start, keys), atol=1e-5)
    assert np.isfinite(s.value).all()


def test_value_mode_profile_delta(placement):
    keys = goal_keys()
    ds = BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir",
                           value_target_mode="profile_delta")
    s = ds[0]
    assert s.value.shape == (8, len(keys))
    # value[k] = normalize_goal(goal) - normalize_goal(profile(keyframe[k]))
    goal_norm = _norm_profile(s.goal, keys)
    np.testing.assert_allclose(s.value[0], goal_norm - _norm_profile(s.start, keys), atol=1e-5)
    # delta shrinks toward the goal along the chunk (mean |delta| decreases)
    assert np.abs(s.value[-1]).mean() <= np.abs(s.value[0]).mean() + 1e-6


def test_value_mode_rejects_unknown(placement):
    with pytest.raises(ValueError):
        BasePolicyDataset([placement], chunk_size=8, sampling_scheme="multiscale_bidir",
                          value_target_mode="bogus")
