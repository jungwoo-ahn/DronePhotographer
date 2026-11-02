"""Tests for the laptop-side control chain. No MAVSDK, no PX4, no network."""

import asyncio
import math

import pytest

from src.drone_control.commands import HOVER, VelocityCommand
from src.drone_control.conventions import from_mavsdk_body, to_mavsdk_body
from src.drone_control.loop import run
from src.drone_control.policy_client import PolicyClient
from src.drone_control.safety import SafetyLimits, VehicleState, clamp
from src.drone_control.sources import ScriptedSource, SquareSource
from src.drone_control.supervisor import Supervisor


# --- conventions: the three sign flips ------------------------------------------
def test_forward_is_the_only_axis_that_does_not_flip():
    f, r, d, y = to_mavsdk_body(VelocityCommand(vx=2.0))
    assert (f, r, d, y) == (2.0, 0.0, 0.0, 0.0)


def test_our_left_becomes_mavsdk_negative_right():
    _, right, _, _ = to_mavsdk_body(VelocityCommand(vy=1.5))
    assert right == -1.5


def test_our_up_becomes_mavsdk_negative_down():
    _, _, down, _ = to_mavsdk_body(VelocityCommand(vz=1.0))
    assert down == -1.0


def test_our_ccw_yaw_becomes_mavsdk_negative_clockwise():
    *_, yawspeed = to_mavsdk_body(VelocityCommand(yaw_rate=45.0))
    assert yawspeed == -45.0


def test_mavsdk_roundtrip_is_identity():
    c = VelocityCommand(vx=1.0, vy=-2.0, vz=0.5, yaw_rate=-20.0)
    back = from_mavsdk_body(*to_mavsdk_body(c))
    assert (back.vx, back.vy, back.vz, back.yaw_rate) == (c.vx, c.vy, c.vz, c.yaw_rate)


def test_descending_in_our_frame_is_positive_down_for_mavsdk():
    """A real sign error would show up here as a drone that climbs when told to sink."""
    _, _, down, _ = to_mavsdk_body(VelocityCommand(vz=-1.0))
    assert down > 0


# --- freshness -------------------------------------------------------------------
def test_command_expires_after_its_ttl():
    c = VelocityCommand(ttl_s=1.0, received_at=100.0)
    assert c.is_fresh(100.5)
    assert not c.is_fresh(101.5)


def test_malformed_server_reply_degrades_to_hover_not_to_garbage():
    c = VelocityCommand.from_json({"vx": "banana", "yaw_rate": None})
    assert c.is_hover()


def test_stamp_uses_the_supplied_clock_not_wall_time():
    # Regression: stamping with time.monotonic() while the supervisor judged freshness
    # with an injected clock silently disabled the stale-link watchdog.
    assert VelocityCommand().stamp(500.0).received_at == 500.0


# --- safety envelope -------------------------------------------------------------
def test_horizontal_clamp_preserves_direction():
    lim = SafetyLimits(max_horizontal_speed=2.0)
    out, reasons = clamp(VelocityCommand(vx=6.0, vy=8.0), VehicleState(altitude=10), lim)
    assert math.isclose(math.hypot(out.vx, out.vy), 2.0)
    assert math.isclose(out.vy / out.vx, 8.0 / 6.0)      # same heading
    assert reasons


def test_ceiling_blocks_climb_but_still_allows_descent():
    lim = SafetyLimits(max_altitude=10.0)
    up, _ = clamp(VelocityCommand(vz=1.0), VehicleState(altitude=10.0), lim)
    down, _ = clamp(VelocityCommand(vz=-1.0), VehicleState(altitude=10.0), lim)
    assert up.vz == 0.0 and down.vz == -1.0


def test_floor_blocks_descent_but_still_allows_climb():
    lim = SafetyLimits(min_altitude=2.0)
    down, _ = clamp(VelocityCommand(vz=-1.0), VehicleState(altitude=2.0), lim)
    up, _ = clamp(VelocityCommand(vz=1.0), VehicleState(altitude=2.0), lim)
    assert down.vz == 0.0 and up.vz == 1.0


def test_geofence_blocks_translation_but_leaves_yaw_so_it_can_turn_around():
    lim = SafetyLimits(max_radius=10.0)
    out, reasons = clamp(VelocityCommand(vx=1.0, yaw_rate=20.0),
                         VehicleState(altitude=5, north=10.0, east=0.0), lim)
    assert out.vx == 0.0 and out.yaw_rate == 20.0
    assert any("geofence" in r for r in reasons)


def test_bad_telemetry_stops_translation():
    out, reasons = clamp(VelocityCommand(vx=1.0, vz=1.0), VehicleState(ok=False), SafetyLimits())
    assert out.vx == 0.0 and out.vz == 0.0
    assert any("telemetry" in r for r in reasons)


# --- the chain of order ----------------------------------------------------------
def _sup(t, **kw):
    return Supervisor(clock=lambda: t[0], **kw)


def test_hovers_before_any_command_arrives():
    t = [0.0]
    assert _sup(t).decide(VehicleState(altitude=5)).command.is_hover()


def test_fresh_command_passes_through():
    t = [0.0]; s = _sup(t)
    s.submit(VelocityCommand(vx=1.0, seq=3))
    d = s.decide(VehicleState(altitude=5))
    assert d.command.vx == 1.0 and not d.stale


def test_lost_policy_link_hovers_once_the_ttl_expires():
    """The case MAVSDK hides: it would keep resending the last setpoint at 20 Hz."""
    t = [0.0]; s = _sup(t)
    s.submit(VelocityCommand(vx=3.0, ttl_s=1.0))
    assert s.decide(VehicleState(altitude=5)).command.vx == 3.0
    t[0] = 1.5                                   # link died, nothing new arrived
    d = s.decide(VehicleState(altitude=5))
    assert d.stale and d.command.is_hover()


def test_a_single_dropped_fetch_does_not_stop_the_drone():
    t = [0.0]; s = _sup(t)
    s.submit(VelocityCommand(vx=2.0, ttl_s=1.0))
    s.submit(None)                               # one failed HTTP call
    t[0] = 0.5
    assert s.decide(VehicleState(altitude=5)).command.vx == 2.0


def test_estop_outranks_a_fresh_command_and_is_latched():
    t = [0.0]; s = _sup(t)
    s.estop()
    s.submit(VelocityCommand(vx=5.0))            # arrives after the stop
    d = s.decide(VehicleState(altitude=5))
    assert d.command.is_hover() and d.should_land
    assert s.decide(VehicleState(altitude=5)).estopped   # still latched


def test_estop_clears_only_explicitly():
    t = [0.0]; s = _sup(t)
    s.estop(); s.clear_estop()
    s.submit(VelocityCommand(vx=1.0))
    assert s.decide(VehicleState(altitude=5)).command.vx == 1.0


def test_envelope_applies_to_policy_commands():
    t = [0.0]
    s = Supervisor(SafetyLimits(max_horizontal_speed=1.0), clock=lambda: t[0])
    s.submit(VelocityCommand(vx=9.0))
    d = s.decide(VehicleState(altitude=5))
    assert d.command.vx == 1.0 and d.clamped


# --- sources ---------------------------------------------------------------------
def test_square_alternates_legs_and_turns_and_turns_left():
    s = SquareSource(speed=1.0, leg_s=2.0, turn_s=1.0, turn_deg_s=30.0)
    assert s.next_command(None, 0.0).vx == 1.0
    assert s.next_command(None, 2.5).yaw_rate == 30.0      # LEFT-positive => CCW
    assert s.next_command(None, 3.5).vx == 1.0


def test_scripted_source_returns_none_when_exhausted():
    s = ScriptedSource([(1.0, VelocityCommand(vx=1.0))])
    assert s.next_command(None, 0.5).vx == 1.0
    assert s.next_command(None, 2.0) is None


def test_policy_client_never_raises_into_the_loop():
    class Dead:
        def post(self, *a, **k):
            raise ConnectionError("tunnel down")
    c = PolicyClient(session=Dead())
    assert c.request_command(b"x", VehicleState(), {}) is None
    assert c.failures == 1


def test_policy_client_parses_a_good_reply():
    class Ok:
        def post(self, *a, **k):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"vx": 1.0, "yaw_rate": -10.0, "note": "hi"}
            return R()
    c = PolicyClient(session=Ok())
    cmd = c.request_command(None, VehicleState(), {})
    assert cmd.vx == 1.0 and cmd.yaw_rate == -10.0 and cmd.source == "policy"


# --- the loop end to end ---------------------------------------------------------
class FakeDriver:
    """Records what the airframe would have received."""
    def __init__(self):
        self.sent = []
        self.landed = False

    async def state(self):
        return VehicleState(altitude=5.0, ok=True)

    async def send(self, decision):
        self.sent.append(to_mavsdk_body(decision.command))


def test_loop_sends_at_control_rate_and_stops_on_duration():
    drv = FakeDriver()
    t = [0.0]
    def clock():
        t[0] += 0.05          # one tick per call, 20 Hz
        return t[0]
    sup = Supervisor(clock=lambda: t[0])
    asyncio.run(run(drv, sup, ScriptedSource([(10.0, VelocityCommand(vx=1.0))]),
                    policy_hz=2.0, duration_s=1.0, log=lambda *a: None, clock=clock))
    assert len(drv.sent) > 5
    assert any(s[0] > 0 for s in drv.sent)      # it really did command forward motion


def test_loop_keeps_flying_when_the_source_raises():
    class Angry:
        name = "angry"
        def next_command(self, state, now):
            raise RuntimeError("policy exploded")
    drv = FakeDriver()
    t = [0.0]
    def clock():
        t[0] += 0.05
        return t[0]
    sup = Supervisor(clock=lambda: t[0])
    asyncio.run(run(drv, sup, Angry(), policy_hz=5.0, duration_s=0.5,
                    log=lambda *a: None, clock=clock))
    assert drv.sent and all(s == (0.0, 0.0, 0.0, 0.0) for s in drv.sent)   # hovered


def test_loop_treats_telemetry_failure_as_unsafe():
    class NoTelem(FakeDriver):
        async def state(self):
            raise RuntimeError("no telemetry")
    drv = NoTelem()
    t = [0.0]
    def clock():
        t[0] += 0.05
        return t[0]
    sup = Supervisor(clock=lambda: t[0])
    asyncio.run(run(drv, sup, ScriptedSource([(10.0, VelocityCommand(vx=5.0))]),
                    policy_hz=5.0, duration_s=0.5, log=lambda *a: None, clock=clock))
    assert all(s[0] == 0.0 for s in drv.sent)
