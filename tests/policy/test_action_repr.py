"""Unit tests for src/policy/common/action_repr.py."""

import numpy as np

from src.policy.common.action_repr import (
    ACTION_DIM,
    ACTION_SCALE,
    apply_action_5d,
    decode_action_5d,
    denormalize_action_5d,
    encode_action_5d,
    normalize_action_5d,
    rotation_matrix_from_yaw_pitch,
    yaw_pitch_from_rotation_matrix,
)


def test_yaw_pitch_round_trip():
    for yaw in [-1.0, -0.3, 0.0, 0.4, 1.2]:
        for pitch in [-0.8, 0.0, 0.6]:
            r = rotation_matrix_from_yaw_pitch(yaw, pitch)
            y2, p2 = yaw_pitch_from_rotation_matrix(r)
            assert abs(yaw - y2) < 1e-5
            assert abs(pitch - p2) < 1e-5


def test_identity_action_for_zero_delta():
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a = encode_action_5d(pos, fwd, up, pos, fwd, up)
    assert a.shape == (ACTION_DIM,)
    np.testing.assert_allclose(a, np.zeros(ACTION_DIM), atol=1e-6)


def test_translation_only_encoded_in_camera_local_basis():
    # Camera looking down +Y world, with +Z world as up.
    pos = np.zeros(3, dtype=np.float32)
    fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Move 5 units to the right (world +X): in camera-local that's +right
    next_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    a = encode_action_5d(pos, fwd, up, next_pos, fwd, up)
    np.testing.assert_allclose(a[:3], [5.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(a[3:], [0.0, 0.0], atol=1e-6)


def test_encode_decode_round_trip_for_pure_rotation():
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fwd = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # rotate camera by yaw=0.3, pitch=-0.2 in camera-local
    rot = rotation_matrix_from_yaw_pitch(0.3, -0.2)
    # new basis: apply rot to camera basis. Same procedure as apply_action_5d
    action = np.array([0.0, 0.0, 0.0, 0.3, -0.2], dtype=np.float32)
    nxt_pos, nxt_fwd, nxt_up = apply_action_5d(pos, fwd, up, action)
    a2 = encode_action_5d(pos, fwd, up, nxt_pos, nxt_fwd, nxt_up)
    np.testing.assert_allclose(a2, action, atol=1e-4)


def test_decode_rejects_wrong_dim():
    try:
        decode_action_5d(np.zeros(4, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_action_scale_shape():
    assert ACTION_SCALE.shape == (ACTION_DIM,)
    assert (ACTION_SCALE > 0).all()


def test_normalize_denormalize_round_trip_within_range():
    # An action comfortably within p99 scale should survive the round trip
    a = np.array([0.1, -0.1, 0.4, 0.04, -0.03], dtype=np.float32)
    n = normalize_action_5d(a)
    back = denormalize_action_5d(n)
    np.testing.assert_allclose(back, a, atol=1e-5)


def test_normalize_action_clips_beyond_scale():
    # 10 m forward dolly >> 0.86 scale → clips to +1
    a = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    n = normalize_action_5d(a)
    assert n[2] == 1.0


def test_normalize_action_maps_scale_to_unit():
    n = normalize_action_5d(ACTION_SCALE.copy())
    np.testing.assert_allclose(n, np.ones(ACTION_DIM), atol=1e-6)


def test_normalize_action_custom_scale():
    a = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    custom = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(normalize_action_5d(a, custom), np.ones(ACTION_DIM), atol=1e-6)


def test_apply_action_5d_translation_in_local_frame():
    pos = np.zeros(3, dtype=np.float32)
    # Camera facing -Z world, up = +Y world (default Blender forward = -Z)
    fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Local "forward" (+Z local in camera basis = forward axis in our basis) => move 2 units forward
    # In the basis matrix from make_camera_basis_from_forward_up, axes are
    # [right, up, forward] in world. forward(=local +Z basis col) is world fwd here = -Z.
    action = np.array([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32)  # +forward 2m
    nxt_pos, _, _ = apply_action_5d(pos, fwd, up, action)
    # Position should move in the +forward direction (world -Z by 2)
    np.testing.assert_allclose(nxt_pos, [0.0, 0.0, -2.0], atol=1e-5)
