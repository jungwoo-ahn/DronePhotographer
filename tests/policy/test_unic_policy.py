"""UNICPolicy box->action mapping tests with a mock recommender (no UNIC weights)."""

import math

import numpy as np
import pytest

from src.policy.common.action_repr import ACTION_DIM
from src.policy.unic.model import UNICRecommendation
from src.policy.unic.policy import UNICPolicy


class _MockUNIC:
    def __init__(self, rec: UNICRecommendation):
        self._rec = rec

    def recommend(self, image):
        return self._rec


def _policy(rec, **kw):
    return UNICPolicy(_MockUNIC(rec), **kw)


def test_centered_full_frame_gives_near_zero_action():
    # box centered, width 1.0 -> no pan, no zoom
    rec = UNICRecommendation(0.5, 0.5, 1.0, 1.0, 0.9)
    a, info = _policy(rec).act(object())
    assert a.shape == (ACTION_DIM,)
    np.testing.assert_allclose(a, np.zeros(ACTION_DIM), atol=1e-6)
    assert info["recommendation"]["score"] == 0.9


def test_box_right_pans_yaw_right():
    rec = UNICRecommendation(0.75, 0.5, 1.0, 1.0, 0.9)
    a, _ = _policy(rec, hfov_deg=50.0).act(object())
    # dx=0.25 -> yaw = 0.25 * 50deg
    assert a[3] == pytest.approx(math.radians(0.25 * 50.0), abs=1e-5)
    assert a[3] > 0 and a[4] == pytest.approx(0.0, abs=1e-6)


def test_box_below_pitches_down():
    rec = UNICRecommendation(0.5, 0.8, 1.0, 1.0, 0.9)
    a, _ = _policy(rec).act(object())
    assert a[4] < 0           # below center -> pitch down (negative)
    assert a[3] == pytest.approx(0.0, abs=1e-6)


def test_tight_crop_dollies_forward_wide_backs_off():
    fwd, _ = _policy(UNICRecommendation(0.5, 0.5, 0.5, 0.5, 0.9), dolly_gain=2.0).act(object())
    back, _ = _policy(UNICRecommendation(0.5, 0.5, 1.5, 1.5, 0.9), dolly_gain=2.0).act(object())
    assert fwd[2] > 0          # width<1 -> forward
    assert back[2] < 0         # width>1 -> back off
    assert fwd[2] == pytest.approx(2.0 * (1.0 - 0.5), abs=1e-5)


def test_action_is_clamped():
    rec = UNICRecommendation(5.0, -5.0, 0.0, 1.0, 0.9)  # absurd offsets / very tight
    a, _ = _policy(rec, max_translation_m=1.0, max_angle_deg=30.0, dolly_gain=100.0).act(object())
    ang = math.radians(30.0)
    assert -1.0 <= a[2] <= 1.0
    assert -ang - 1e-6 <= a[3] <= ang + 1e-6
    assert -ang - 1e-6 <= a[4] <= ang + 1e-6


def test_no_lateral_translation():
    rec = UNICRecommendation(0.7, 0.3, 0.6, 0.6, 0.9)
    a, _ = _policy(rec).act(object())
    assert a[0] == 0.0 and a[1] == 0.0   # pan is realized as rotation, not lateral move
