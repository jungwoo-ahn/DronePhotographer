"""The 10D pose action, split between the airframe and the gimbal."""

import math

import numpy as np
import pytest

from src.policy.common.action_to_drone import TILT_MAX_DEG, TILT_MIN_DEG, action_to_move
from src.policy.common.action_repr import encode_action_9d


def _aim(az_deg, el_deg):
    """A roll-free camera pose at this world azimuth and elevation."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    return forward.astype(np.float32), up.astype(np.float32)


def _action(az0, el0, az1, el1, delta_world=(0.0, 0.0, 0.0)):
    f0, u0 = _aim(az0, el0)
    f1, u1 = _aim(az1, el1)
    p0 = np.zeros(3, dtype=np.float32)
    return encode_action_9d(p0, f0, u0, np.asarray(delta_world, dtype=np.float32), f1, u1)


def test_no_motion_stays_put():
    m = action_to_move(_action(0, -20, 0, -20), tilt_deg=-20, dt=1.0)
    for v in (m.vx, m.vy, m.vz, m.yaw_rate, m.tilt_rate):
        assert abs(v) < 1e-3
    assert m.tilt_target == pytest.approx(-20, abs=1e-3) and not m.shoot


def test_elevation_change_becomes_tilt_alone():
    m = action_to_move(_action(0, -10, 0, -25), tilt_deg=-10, dt=2.0)
    assert m.tilt_target == pytest.approx(-25, abs=1e-2)
    assert m.tilt_rate == pytest.approx(-7.5, abs=1e-2)
    assert abs(m.yaw_rate) < 1e-2


def test_azimuth_change_becomes_yaw_alone_and_left_is_positive():
    m = action_to_move(_action(0, -30, 20, -30), tilt_deg=-30, dt=1.0)
    assert m.yaw_rate == pytest.approx(20, abs=1e-2)
    assert abs(m.tilt_rate) < 1e-2


def test_translation_uses_the_tilted_camera_frame():
    """Forward along a camera looking 30° down is forward AND down for the aircraft."""
    down30 = (math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30)))
    m = action_to_move(_action(0, -30, 0, -30, delta_world=down30), tilt_deg=-30, dt=1.0)
    assert m.vx == pytest.approx(down30[0], abs=1e-4)
    assert m.vz == pytest.approx(down30[2], abs=1e-4)
    assert abs(m.vy) < 1e-4


def test_strafe_left_is_positive_vy():
    m = action_to_move(_action(0, -30, 0, -30, delta_world=(0.0, 0.5, 0.0)), tilt_deg=-30, dt=1.0)
    assert m.vy == pytest.approx(0.5, abs=1e-4) and abs(m.vx) < 1e-4 and abs(m.vz) < 1e-4


@pytest.mark.parametrize("el0,el1,az,delta", [(-45, -20, 35, (0.3, -0.2, 0.1)),
                                              (10, 20, -60, (-0.1, 0.05, -0.2)),
                                              (-70, -80, 5, (0.0, 0.4, 0.3))])
def test_any_level_move_comes_back_exactly(el0, el1, az, delta):
    m = action_to_move(_action(0, el0, az, el1, delta_world=delta), tilt_deg=el0, dt=0.5)
    assert (m.vx * 0.5, m.vy * 0.5, m.vz * 0.5) == pytest.approx(delta, abs=1e-3)
    assert m.yaw_rate * 0.5 == pytest.approx(az, abs=2e-2)
    assert m.tilt_target == pytest.approx(el1, abs=2e-2)


def test_angles_past_the_gimbal_are_limited_and_reported():
    m = action_to_move(_action(0, 20, 0, 40), tilt_deg=20, dt=1.0)
    assert m.tilt_target == TILT_MAX_DEG and m.tilt_limited
    assert m.tilt_rate == pytest.approx(TILT_MAX_DEG - 20)
    assert TILT_MIN_DEG == -90.0


def test_shoot_channel_and_a_bad_dt():
    a = np.concatenate([_action(0, 0, 0, 0), [1.0]])
    assert action_to_move(a, tilt_deg=0, dt=1.0).shoot
    with pytest.raises(ValueError):
        action_to_move(a, tilt_deg=0, dt=0.0)
