"""Pin down the 5D action's sign conventions.

These were wrong in two places at once: the GPT-5 agent's prompt and UNIC's pan mapping both
assumed "positive dyaw turns right", while `apply_action_5d` yaws about WORLD up in the
left-positive direction. Every yaw was inverted, which made both policies score worse than
standing still. The conventions are not self-evident from the code, so they get asserted here.
"""

import numpy as np
import pytest

from src.policy.common.action_repr import (POSE_DIM_5D, apply_action_5d,
                                           yaw_from_right_positive)
from src.utils.rotation_utils import make_camera_basis_from_forward_up

POS = np.array([0.0, 0.0, 0.0], np.float32)
FWD = np.array([1.0, 0.0, 0.0], np.float32)
UP = np.array([0.0, 0.0, 1.0], np.float32)


def _act(**kw):
    a = np.zeros(POSE_DIM_5D, np.float32)
    for i, k in enumerate(("dx", "dy", "dz", "dyaw", "dpitch")):
        a[i] = kw.get(k, 0.0)
    return apply_action_5d(POS, FWD, UP, a)


def _basis():
    b = make_camera_basis_from_forward_up(FWD, UP)
    return b[:, 0], b[:, 1]          # right, up


def test_translation_axes_are_right_up_forward():
    right, up = _basis()
    assert np.allclose(_act(dx=1.0)[0], right, atol=1e-6)
    assert np.allclose(_act(dy=1.0)[0], up, atol=1e-6)
    assert np.allclose(_act(dz=1.0)[0], FWD, atol=1e-6)


def test_raw_dyaw_is_left_positive():
    """The surprising one: +dyaw swings `forward` AWAY from the camera's right axis."""
    right, _ = _basis()
    assert float(np.dot(_act(dyaw=0.2)[1], right)) < -0.1
    assert float(np.dot(_act(dyaw=-0.2)[1], right)) > 0.1


def test_yaw_from_right_positive_turns_right():
    """A right-positive angle, converted, must actually turn the camera right."""
    right, _ = _basis()
    assert float(np.dot(_act(dyaw=yaw_from_right_positive(0.2))[1], right)) > 0.1
    assert float(np.dot(_act(dyaw=yaw_from_right_positive(-0.2))[1], right)) < -0.1


def test_dpitch_is_up_positive():
    _, up = _basis()
    assert float(np.dot(_act(dpitch=0.2)[1], up)) > 0.1
    assert float(np.dot(_act(dpitch=-0.2)[1], up)) < -0.1


def test_unic_pan_recentres_the_recommended_box():
    """UNIC's whole pan mapping: a box right of centre must induce a turn to the right."""
    from src.policy.unic.model import UNICRecommendation
    from src.policy.unic.policy import UNICPolicy

    class _Mock:
        def __init__(self, cx):
            self.cx = cx

        def recommend(self, image):
            return UNICRecommendation(self.cx, 0.5, 0.8, 0.8, 1.0)

    right, _ = _basis()
    for cx, expect_right in ((0.8, True), (0.2, False)):
        a, _ = UNICPolicy(_Mock(cx)).act(None)
        nf = apply_action_5d(POS, FWD, UP, a)[1]
        d = float(np.dot(nf, right))
        assert (d > 0.01) is expect_right, f"cx={cx} turned the wrong way (along-right {d:+.3f})"


@pytest.mark.parametrize("bad", [np.zeros(9, np.float32), np.zeros(10, np.float32)])
def test_9d_actions_are_rejected_by_the_5d_path(bad):
    with pytest.raises(ValueError):
        apply_action_5d(POS, FWD, UP, bad)
