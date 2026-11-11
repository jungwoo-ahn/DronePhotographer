"""Tests for the laptop-side control chain. No MAVSDK, no PX4, no network."""

import asyncio
import math
import time
from pathlib import Path

import pytest

from src.drone_control.commands import HOVER, VelocityCommand
from src.drone_control.conventions import from_mavsdk_body, to_mavsdk_body
from src.drone_control.loop import run
from src.drone_control.policy_client import PolicyClient
from src.drone_control.safety import SafetyLimits, VehicleState, clamp
from src.drone_control.sources import HoverSource, ScriptedSource, SquareSource
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


# --- the camera channel (second, independent link) -------------------------------
from src.drone_control.camera import CameraStream, Frame          # noqa: E402


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in."""
    def __init__(self, frames, fail_after=None):
        self.frames, self.i, self.fail_after = frames, 0, fail_after
        self.released = False

    def isOpened(self): return True

    def read(self):
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        if self.i >= len(self.frames):
            time.sleep(0.01)
            return True, self.frames[-1]
        f = self.frames[self.i]; self.i += 1
        return True, f

    def release(self): self.released = True


def test_camera_keeps_only_the_newest_frame():
    import time as _t
    cam = CameraStream("x", open_capture=lambda u: FakeCapture(["a", "b", "c"]), settle_s=0.0)
    cam.start()
    deadline = _t.time() + 2.0
    while _t.time() < deadline and cam.frames_read < 3:
        _t.sleep(0.01)
    cam.stop()
    assert cam.latest().image == "c"          # not "a" -- no backlog
    assert cam.frames_read >= 3


def test_frames_right_after_connecting_are_dropped():
    now = [0.0]
    seen = {}
    capture = FakeCapture(["frame"])
    read = capture.read

    def read_one_second_later():
        seen[now[0]] = cam.latest()
        now[0] += 1.0
        return read()

    capture.read = read_one_second_later
    cam = CameraStream("x", open_capture=lambda u: capture, clock=lambda: now[0], settle_s=3.0)
    cam.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and now[0] < 6.0:
        time.sleep(0.01)
    cam.stop()
    assert all(seen[t] is None for t in (0.0, 1.0, 2.0))    # the read that returns at 3 s is kept
    assert seen[3.0] is not None


def test_stale_frame_is_withheld():
    cam = CameraStream("x", max_age_s=0.5, open_capture=lambda u: FakeCapture(["a"]))
    cam._frame = Frame(image="a", seq=1, received_at=100.0)
    assert cam.fresh_frame(now=100.2) is not None
    assert cam.fresh_frame(now=101.0) is None


def test_camera_reconnects_when_the_stream_drops():
    import time as _t
    cam = CameraStream("x", reconnect_delay_s=0.01,
                       open_capture=lambda u: FakeCapture(["a"], fail_after=1))
    cam.start()
    deadline = _t.time() + 2.0
    while _t.time() < deadline and cam.reconnects < 2:
        _t.sleep(0.01)
    cam.stop()
    assert cam.reconnects >= 2


class _FixedCam:
    def __init__(self, frame): self._f = frame
    def fresh_frame(self, now=None): return self._f


# --- the ground-station panel ----------------------------------------------------
import json as _json                                              # noqa: E402
import urllib.request as _url                                     # noqa: E402

from src.drone_control.ui import UiServer, UiState                # noqa: E402


def _decide(sup, st):
    return sup.decide(st)


def test_ui_snapshot_shows_both_frames_so_a_sign_flip_is_visible():
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    sup.submit(VelocityCommand(vy=1.0, yaw_rate=12.0))
    st = VehicleState(altitude=5.0, north=3.0, east=4.0)
    ui = UiState(SafetyLimits())
    ui.publish(decision=_decide(sup, st), state=st, now=1.0, started_at=0.0)
    s = ui.snapshot()
    assert s["command"]["vy"] == 1.0 and s["sent"]["right"] == -1.0
    assert s["command"]["yaw_rate"] == 12.0 and s["sent"]["yawspeed"] == -12.0
    assert s["telemetry"]["radius"] == pytest.approx(5.0)


def test_ui_reports_the_active_rung():
    t = [0.0]
    st = VehicleState(altitude=5.0)
    sup = Supervisor(SafetyLimits(max_horizontal_speed=1.0), clock=lambda: t[0])
    ui = UiState(SafetyLimits(max_horizontal_speed=1.0))

    ui.publish(decision=_decide(sup, st), state=st, now=0.0)
    assert ui.snapshot()["rung"] == "idle"

    sup.submit(VelocityCommand(vx=9.0, ttl_s=1.0))
    ui.publish(decision=_decide(sup, st), state=st, now=0.1)
    assert ui.snapshot()["rung"] == "envelope"

    t[0] = 5.0
    ui.publish(decision=_decide(sup, st), state=st, now=5.0)
    assert ui.snapshot()["rung"] == "stale"

    sup.estop()
    ui.publish(decision=_decide(sup, st), state=st, now=5.1)
    assert ui.snapshot()["rung"] == "estop"


def test_ui_logs_only_transitions_not_every_tick():
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    st = VehicleState(altitude=5.0)
    ui = UiState(SafetyLimits())
    for i in range(10):
        ui.publish(decision=_decide(sup, st), state=st, now=float(i))
    assert len(ui.snapshot()["events"]) == 1      # ten identical ticks, one event


def _serve(sup, ui, port):
    srv = UiServer(ui, sup, port=port).start()
    return srv


def test_ui_server_serves_assets_and_state():
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    st = VehicleState(altitude=5.0)
    ui = UiState(SafetyLimits())
    ui.publish(decision=_decide(sup, st), state=st, now=0.0)
    srv = _serve(sup, ui, 8123)
    try:
        for path, ctype in (("/", "text/html"), ("/style.css", "text/css"),
                            ("/app.js", "application/javascript")):
            r = _url.urlopen(f"http://127.0.0.1:8123{path}", timeout=5)
            assert r.status == 200 and r.headers["Content-Type"] == ctype
            assert len(r.read()) > 100
        s = _json.load(_url.urlopen("http://127.0.0.1:8123/api/state", timeout=5))
        assert s["rung"] == "idle"
    finally:
        srv.stop()


def test_ui_estop_endpoint_actually_stops_the_supervisor():
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    sup.submit(VelocityCommand(vx=2.0))
    ui = UiState(SafetyLimits())
    srv = _serve(sup, ui, 8124)
    try:
        _url.urlopen(_url.Request("http://127.0.0.1:8124/api/estop", method="POST"), timeout=5)
        d = sup.decide(VehicleState(altitude=5.0))
        assert d.estopped and d.command.is_hover()
        _url.urlopen(_url.Request("http://127.0.0.1:8124/api/clear_estop", method="POST"),
                     timeout=5)
        assert not sup.decide(VehicleState(altitude=5.0)).estopped
    finally:
        srv.stop()


def test_ui_offers_no_way_to_command_the_drone():
    """The panel observes the chain of order; it must not become another rung in it."""
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    ui = UiState(SafetyLimits())
    srv = _serve(sup, ui, 8125)
    try:
        req = _url.Request("http://127.0.0.1:8125/api/velocity", method="POST", data=b"{}")
        with pytest.raises(Exception) as e:
            _url.urlopen(req, timeout=5)
        assert "404" in str(e.value)
    finally:
        srv.stop()


def test_ui_frame_endpoint_returns_204_without_a_camera():
    t = [0.0]
    sup = Supervisor(clock=lambda: t[0])
    ui = UiState(SafetyLimits())
    srv = _serve(sup, ui, 8126)
    try:
        r = _url.urlopen("http://127.0.0.1:8126/api/frame.jpg", timeout=5)
        assert r.status == 204
    finally:
        srv.stop()


def test_ui_failure_cannot_stop_the_control_loop():
    """A broken panel must never be able to ground the aircraft."""
    class Exploding:
        def publish(self, **kw):
            raise RuntimeError("ui blew up")
    drv = FakeDriver()
    t = [0.0]
    def clock():
        t[0] += 0.05
        return t[0]
    sup = Supervisor(clock=lambda: t[0])
    asyncio.run(run(drv, sup, ScriptedSource([(10.0, VelocityCommand(vx=1.0))]),
                    policy_hz=5.0, duration_s=0.5, log=lambda *a: None,
                    clock=clock, ui=Exploding()))
    assert drv.sent and any(s[0] > 0 for s in drv.sent)   # still flew


# --- flying move by move: the policy proposes, the operator runs it ---------------
from src.drone_control.approval import ApprovalGate, describe_move       # noqa: E402


class _OneShot:
    """A source that keeps proposing the same move."""
    name = "oneshot"

    def __init__(self, cmd):
        self.cmd = cmd
        self.asks = 0

    def next_command(self, state, now):
        self.asks += 1
        return self.cmd


def _gate(cmd=None, **kw):
    t = [0.0]
    src = _OneShot(cmd or VelocityCommand(vy=0.9, yaw_rate=8.0))
    return ApprovalGate(src, clock=lambda: t[0], **kw), src, t


def test_nothing_flies_until_someone_accepts():
    gate, src, t = _gate()
    st = VehicleState(altitude=5.0, ok=True)
    assert gate.next_command(st, 0.0).is_hover()     # proposes, commands a hover
    assert gate.snapshot(0.0)["offer"] is not None
    assert gate.next_command(st, 0.1).is_hover()     # still holding, offer still up
    gate.accept(now=0.2)
    assert not gate.next_command(st, 0.3).is_hover()  # now it flies


def test_an_accepted_move_ends_on_the_clock_not_on_a_second_click():
    gate, src, t = _gate(hold_s=1.0)
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    gate.accept(now=0.0)
    assert not gate.next_command(st, 0.9).is_hover()
    assert gate.next_command(st, 1.1).is_hover()     # the clock ended it
    assert gate.snapshot(1.1)["record"][0]["outcome"] == "flown"


def test_accepting_releases_exactly_what_the_policy_proposed():
    proposed = VelocityCommand(vx=0.4, vy=0.9, vz=0.0, yaw_rate=8.0)
    gate, src, t = _gate(proposed)
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    gate.accept(now=0.0)
    out = gate.next_command(st, 0.1)
    assert (out.vx, out.vy, out.vz, out.yaw_rate) == (0.4, 0.9, 0.0, 8.0)


def test_accepting_nothing_does_nothing():
    gate, src, t = _gate()
    assert gate.accept(now=0.0) is False
    assert gate.next_command(VehicleState(ok=True), 0.0).is_hover()
    assert gate.snapshot(0.0)["running"] is None


def test_a_move_nobody_runs_is_withdrawn_rather_than_left_on_offer():
    gate, src, t = _gate(offer_s=2.0)
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    assert gate.snapshot(0.0)["offer"] is not None
    gate.next_command(st, 3.0)                       # past the offer window
    assert gate.expired == 1
    assert gate.snapshot(3.0)["record"][0]["outcome"] == "expired"


def test_skipping_asks_the_policy_again_from_a_newer_view():
    gate, src, t = _gate()
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    asks = src.asks
    assert gate.skip() is True
    gate.next_command(st, 0.5)
    assert src.asks == asks + 1


def test_the_offer_shows_what_the_envelope_will_do_to_it_before_you_click():
    gate, src, t = _gate(VelocityCommand(vy=9.0),
                         limits=SafetyLimits(max_horizontal_speed=1.0))
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    offer = gate.snapshot(0.0)["offer"]
    assert offer["asked"]["vy"] == 9.0
    assert offer["command"]["vy"] == 1.0             # what will actually fly
    assert offer["clamped"]                          # and it says so
    assert offer["sent"]["right"] == -1.0            # still converted for MAVSDK


def test_a_dead_panel_cannot_leave_a_move_running():
    """The click starts it; only the clock stops it. No second request is needed."""
    gate, src, t = _gate(hold_s=0.5)
    st = VehicleState(altitude=5.0, ok=True)
    gate.next_command(st, 0.0)
    gate.accept(now=0.0)
    # the browser now vanishes: no further accept, no skip, no poll
    assert gate.next_command(st, 0.6).is_hover()


def test_the_move_is_described_in_words_an_operator_can_check():
    assert describe_move(VelocityCommand(vy=0.9, yaw_rate=8.0)) == "Arc left around the subject"
    assert describe_move(VelocityCommand(vx=0.4, vy=0.9, yaw_rate=8.0)) == \
        "Arc left around the subject, moving closer"
    assert describe_move(VelocityCommand(vy=-0.9, yaw_rate=-8.0)) == "Arc right around the subject"
    assert describe_move(VelocityCommand(vz=0.5)) == "Climb"
    assert describe_move(VelocityCommand()) == "Hold still"
    assert "closer" in describe_move(VelocityCommand(vx=0.4))


def test_the_real_policys_small_steps_are_still_called_moves():
    # A step the real model gave: about 5 cm over 1.5 s, which once read "Hold still".
    assert describe_move(VelocityCommand(vx=-0.05, vy=0.03, vz=-0.02)) == \
        "Slide left, moving back and descending"
    # A small part next to a big one goes unnamed, so the sentence says what matters.
    assert describe_move(VelocityCommand(vx=0.4, vz=0.02)) == "Move closer"
    assert describe_move(VelocityCommand(vx=0.004, yaw_rate=0.2)) == "Hold still"


def test_the_panel_can_run_and_skip_but_still_cannot_author_a_command():
    gate, src, t = _gate()
    sup = Supervisor(clock=lambda: 0.0)
    ui = UiState(SafetyLimits())
    gate.next_command(VehicleState(altitude=5.0, ok=True), 0.0)
    srv = UiServer(ui, sup, port=8127, gate=gate).start()
    try:
        base = "http://127.0.0.1:8127"
        r = _json.load(_url.urlopen(_url.Request(base + "/api/accept", method="POST"), timeout=5))
        assert r["ran"] is True
        with pytest.raises(Exception):
            _url.urlopen(_url.Request(base + "/api/velocity", method="POST", data=b"{}"),
                         timeout=5)
    finally:
        srv.stop()


def test_a_latched_stop_outranks_the_run_button():
    """Clicking Run while stopped must not consume the move or write a false record."""
    gate, src, t = _gate()
    sup = Supervisor(clock=lambda: 0.0)
    sup.estop()
    ui = UiState(SafetyLimits())
    gate.next_command(VehicleState(altitude=5.0, ok=True), 0.0)
    srv = UiServer(ui, sup, port=8128, gate=gate).start()
    try:
        r = _json.load(_url.urlopen(
            _url.Request("http://127.0.0.1:8128/api/accept", method="POST"), timeout=5))
        assert r["ran"] is False
        assert gate.snapshot(0.0)["offer"] is not None      # still on offer
        assert gate.snapshot(0.0)["running"] is None
    finally:
        srv.stop()


# --- real hardware: a slow or silent link, and refusing on the ground ------------
def test_silent_telemetry_cannot_freeze_the_estop():
    """One silent radio used to hang the loop, with the stop and the TTL behind it."""
    class Silent(FakeDriver):
        async def state(self):
            await asyncio.Event().wait()

    async def go():
        drv, sup = Silent(), Supervisor()
        task = asyncio.ensure_future(run(drv, sup, ScriptedSource([(60.0, VelocityCommand(vx=1.0))]),
                                         duration_s=30.0, log=lambda *a: None))
        await asyncio.sleep(0.1)
        sup.estop()
        await asyncio.wait_for(task, 2.0)       # raises if the loop never comes round
        assert drv.sent[-1] == (0.0, 0.0, 0.0, 0.0)
    asyncio.run(go())


def test_a_hung_send_cannot_freeze_the_estop():
    class Hung(FakeDriver):
        async def send(self, decision):
            await asyncio.Event().wait()

    async def go():
        sup = Supervisor()
        task = asyncio.ensure_future(run(Hung(), sup, HoverSource(), duration_s=30.0,
                                         log=lambda *a: None))
        await asyncio.sleep(0.1)
        sup.estop()
        await asyncio.wait_for(task, 2.0)
    asyncio.run(go())


class _Position:
    def __init__(self, down):
        self.position = type("P", (), {"down_m": down, "north_m": 1.0, "east_m": 2.0})


def _driver_with_one_fix(t):
    from src.drone_control.offboard import OffboardDriver

    async def stream():
        yield _Position(-5.0)
        await asyncio.Event().wait()            # then the radio goes quiet

    drv = OffboardDriver(clock=lambda: t[0])
    drv._drone = type("D", (), {"telemetry": type("T", (), {
        "position_velocity_ned": staticmethod(stream)})})
    return drv


def test_driver_state_never_waits_and_goes_unsafe_when_old():
    from src.drone_control.offboard import TELEMETRY_MAX_AGE_S

    async def go():
        t = [100.0]
        drv = _driver_with_one_fix(t)
        assert (await drv.state()).ok is False          # nothing has arrived yet
        await asyncio.sleep(0)
        fresh = await drv.state()
        assert fresh.ok and fresh.altitude == 5.0 and fresh.east == 2.0
        t[0] += TELEMETRY_MAX_AGE_S + 0.1
        stale = await asyncio.wait_for(drv.state(), 0.1)
        assert stale.ok is False and stale.altitude == 5.0
        drv._watcher.cancel()
    asyncio.run(go())


def test_landing_is_still_sent_when_leaving_offboard_hangs(monkeypatch):
    from src.drone_control import offboard as ob
    monkeypatch.setattr(ob, "LAND_STEP_TIMEOUT_S", 0.05)
    landed = []

    class Plugin:
        async def stop(self):
            await asyncio.Event().wait()

        async def set_velocity_body(self, v):
            pass

        async def land(self):
            landed.append(True)

    drv = ob.OffboardDriver()
    drv._VBY = lambda *a: a
    drv.started = True
    drv._drone = type("D", (), {"offboard": Plugin(), "action": Plugin()})
    asyncio.run(drv.stop_and_land())
    assert landed == [True]


def test_bench_driver_cannot_reach_the_airframe():
    from src.drone_control.offboard import BenchDriver

    class Explodes:
        def __getattr__(self, name):
            raise AssertionError(f"the bench touched the airframe: {name}")

    async def go():
        drv = BenchDriver()
        drv._drone = Explodes()
        await drv.arm_and_takeoff(5.0)
        await drv.start()
        await drv.send(None)
        await drv.hover()
        await drv.stop_and_land()
    asyncio.run(go())


def test_dry_run_flies_in_the_direction_it_was_told():
    from src.drone_control.offboard import DryRunDriver
    from src.drone_control.supervisor import Decision

    async def go():
        t = [0.0]
        drv = DryRunDriver(clock=lambda: t[0])
        await drv.arm_and_takeoff(5.0)
        await drv.start()
        t[0] = 1.0
        await drv.send(Decision(VelocityCommand(vx=2.0), "forward"))
        s = await drv.state()
        assert s.altitude == 5.0 and abs(s.north - 2.0) < 1e-9 and abs(s.east) < 1e-9
    asyncio.run(go())


def test_panel_shows_the_camera_and_encodes_each_frame_once(monkeypatch):
    from src.drone_control.ui import state as ui_state
    encoded = []
    monkeypatch.setattr(ui_state, "encode_jpeg", lambda img: encoded.append(img) or b"jpeg")

    class Cam:
        def __init__(self):
            self.f = None

        def latest(self):
            return self.f

    cam, ui = Cam(), UiState()
    ui.attach_camera(cam)
    assert ui.frame() is None                   # no frame yet: the server answers 204
    cam.f = Frame(image="a", seq=1)
    assert ui.frame() == b"jpeg" and ui.frame() == b"jpeg"
    assert encoded == ["a"]


def _load_pilot():
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_sitl_pilot",
                                                  root / "scripts" / "run_sitl_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pilot_args(**kw):
    base = dict(url="x", source="square", policy_url="http://127.0.0.1:1/act", policy_timeout=0.1,
                policy_hz=2.0, altitude=5.0, takeoff_settle=0.0, speed=1.0, max_vertical_speed=1.0,
                max_yaw_rate=60.0, max_altitude=30.0, max_radius=50.0, duration=0.2,
                camera_url=None, camera_wait=0.2, max_frame_age=0.7, goal_file=None, tilt_deg=0.0,
                lens_hfov=0.0, policy_warmup=1.0, approve=False, server_tunnel=False, gimbal=False, full_hd=False,
                stream_kbps=4096, photo_dir="outputs/photos",
                move_seconds=1.5, ui=False, ui_port=0, bench=False, dry_run=False)
    base.update(kw)
    return type("A", (), base)


def _recording_driver(calls, connect_error=None):
    class Rec:
        def __init__(self, url=None):
            pass

        async def connect(self):
            calls.append("connect")
            if connect_error:
                raise connect_error

        async def wait_until_ready(self):
            pass

        async def arm_and_takeoff(self, a):
            calls.append("takeoff")

        async def start(self):
            pass

        async def send(self, d):
            pass

        async def state(self):
            return VehicleState(altitude=5.0)

        async def stop_and_land(self):
            calls.append("land")
    return Rec


def test_missing_camera_url_is_refused_before_anything_connects(monkeypatch):
    pilot = _load_pilot()
    calls = []
    monkeypatch.setattr(pilot, "OffboardDriver", _recording_driver(calls))
    with pytest.raises(SystemExit):
        asyncio.run(pilot.main_async(_pilot_args(source="policy")))
    assert calls == []


def _goal_file(tmp_path):
    p = tmp_path / "goal.json"
    p.write_text('{"profile": {}}')
    return p


def test_no_camera_frame_means_no_takeoff(monkeypatch, tmp_path):
    pilot = _load_pilot()
    calls = []
    monkeypatch.setattr(pilot, "OffboardDriver", _recording_driver(calls))

    class Dark:
        max_age_s = 0.7

        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def stop(self):
            pass

        def fresh_frame(self):
            return None

    monkeypatch.setattr(pilot, "CameraStream", Dark)
    with pytest.raises(SystemExit):
        asyncio.run(pilot.main_async(_pilot_args(source="policy", camera_url="rtsp://nobody",
                                                 goal_file=str(_goal_file(tmp_path)))))
    assert "takeoff" not in calls


def test_stop_pressed_on_the_ground_means_no_takeoff(monkeypatch):
    pilot = _load_pilot()
    calls = []
    monkeypatch.setattr(pilot, "OffboardDriver", _recording_driver(calls))

    class Stopped(pilot.Supervisor):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.estop()

    monkeypatch.setattr(pilot, "Supervisor", Stopped)
    with pytest.raises(SystemExit):
        asyncio.run(pilot.main_async(_pilot_args()))
    assert "takeoff" not in calls


def test_normal_run_lands_and_a_failed_connection_does_not_pretend_to(monkeypatch, capsys):
    pilot = _load_pilot()
    calls = []
    monkeypatch.setattr(pilot, "OffboardDriver", _recording_driver(calls))
    asyncio.run(pilot.main_async(_pilot_args()))
    assert calls == ["connect", "takeoff", "land"]

    capsys.readouterr()
    monkeypatch.setattr(pilot, "OffboardDriver",
                        _recording_driver([], connect_error=RuntimeError("no vehicle")))
    with pytest.raises(RuntimeError):
        asyncio.run(pilot.main_async(_pilot_args()))
    assert "landing" not in capsys.readouterr().out


def test_dry_run_runs_the_whole_pilot_with_the_panel():
    pilot = _load_pilot()
    asyncio.run(pilot.main_async(_pilot_args(dry_run=True, approve=True, ui=True, ui_port=8131,
                                             duration=0.3)))


# --- the interface: the server sends a sequence, the operator approves each action ---
_AHEAD = [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


class _Client:
    def __init__(self, n=3):
        self.n, self.asked, self.last_error = n, [], None

    def request_actions(self, jpeg, goal, timeout_s=None):
        self.asked.append((jpeg, goal, timeout_s))
        return [list(_AHEAD) for _ in range(self.n)] if self.n else None


class _Cam:
    def __init__(self, frame=True):
        self.frame = Frame(image="img", seq=1) if frame else None

    def fresh_frame(self, now=None):
        return self.frame


def _sequence(monkeypatch, client, cam=None, **kw):
    from src.drone_control import sources
    from src.drone_control.sources import ActionSequenceSource
    monkeypatch.setattr(sources, "encode_jpeg", lambda img, q: b"jpeg")
    return ActionSequenceSource(cam or _Cam(), client, {"profile": {}}, move_s=1.5, **kw)


def _next(src):
    """The next action, allowing for the forward pass on its own thread."""
    import time
    for _ in range(200):
        cmd = src.next_command(None, 0.0)
        if cmd is not None:
            return cmd
        time.sleep(0.01)
    return None


def _offer(gate, state, now):
    """Turn the gate over until something is on offer, as the slow lane does."""
    import time
    for _ in range(200):
        gate.next_command(state, now)
        offer = gate.snapshot(now)["offer"]
        if offer is not None:
            return offer["note"].split(".")[0]
        time.sleep(0.01)
    return None


def test_a_forward_pass_does_not_hold_up_the_loop(monkeypatch):
    import threading
    release = threading.Event()

    class Slow(_Client):
        def request_actions(self, jpeg, goal, timeout_s=None):
            release.wait(5.0)
            return super().request_actions(jpeg, goal, timeout_s)

    src = _sequence(monkeypatch, Slow())
    assert src.next_command(None, 0.0) is None      # returned at once, not after the pass
    release.set()
    assert _next(src) is not None


def test_one_forward_pass_gives_each_action_its_own_turn(monkeypatch):
    client = _Client()
    src = _sequence(monkeypatch, client)
    notes = [_next(src).note for _ in range(3)]
    assert len(client.asked) == 1
    assert [n.split(".")[0] for n in notes] == ["Action 1 of 3", "Action 2 of 3", "Action 3 of 3"]
    _next(src)
    assert len(client.asked) == 2           # asked again only once the sequence ran out


def test_the_action_is_flown_as_the_laptop_works_it_out(monkeypatch):
    cmd = _next(_sequence(monkeypatch, _Client()))
    assert abs(cmd.vx - 0.3 / 1.5) < 1e-9 and cmd.vy == 0.0 and cmd.yaw_rate == 0.0


def test_no_frame_means_the_server_is_not_asked(monkeypatch):
    client = _Client()
    assert _next(_sequence(monkeypatch, client, _Cam(frame=False))) is None
    assert client.asked == []


def test_no_answer_means_no_command(monkeypatch):
    assert _next(_sequence(monkeypatch, _Client(n=0))) is None


def test_each_action_waits_for_its_own_approval(monkeypatch):
    client = _Client()
    t = [0.0]
    gate = ApprovalGate(_sequence(monkeypatch, client), clock=lambda: t[0], hold_s=1.0,
                        offer_s=float("inf"))
    state = VehicleState(altitude=5.0)

    assert _offer(gate, state, t[0]) == "Action 1 of 3"
    t[0] = 1000.0                           # however long the operator takes
    assert gate.next_command(state, t[0]).is_hover()
    assert _offer(gate, state, t[0]) == "Action 1 of 3"
    assert gate.accept()
    assert gate.next_command(state, t[0]).vx > 0
    t[0] += 2.0
    assert gate.next_command(state, t[0]).is_hover()
    assert _offer(gate, state, t[0]) == "Action 2 of 3"     # and nothing moves until approved
    assert len(client.asked) == 1


def test_skipping_drops_the_rest_of_the_sequence(monkeypatch):
    client = _Client()
    t = [0.0]
    gate = ApprovalGate(_sequence(monkeypatch, client), clock=lambda: t[0],
                        offer_s=float("inf"))
    state = VehicleState(altitude=5.0)
    _offer(gate, state, t[0])
    assert gate.skip()
    assert _offer(gate, state, t[0]) == "Action 1 of 3"
    assert len(client.asked) == 2


def test_client_never_raises_and_keeps_what_went_wrong():
    class Dead:
        def post(self, *a, **k):
            raise ConnectionError("tunnel down")
    c = PolicyClient(session=Dead())
    assert c.request_actions(b"x", {}) is None
    assert c.failures == 1 and "tunnel down" in c.last_error


def test_client_treats_a_server_error_as_a_failure():
    class Refusing:
        def post(self, *a, **k):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"actions": [], "error": "KeyError: 'profile'"}
            return R()
    c = PolicyClient(session=Refusing())
    assert c.request_actions(b"x", {}) is None and "profile" in c.last_error


def test_the_stub_server_speaks_the_same_interface():
    """Real HTTP, real handler, through the laptop's client and its conversion."""
    import importlib.util
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    from src.drone_control.actions import action_to_move

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("serve_policy", root / "scripts" / "serve_policy.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    srv = HTTPServer(("127.0.0.1", 0), serve.make_handler(serve.StubPolicy()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Session:
        def post(self, url, json=None, timeout=None):
            req = urllib.request.Request(url, data=_json.dumps(json).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = _json.loads(r.read())

            class R:
                def raise_for_status(self): pass
                def json(self): return body
            return R()

    _json = json
    try:
        c = PolicyClient(url=f"http://127.0.0.1:{srv.server_port}/act", session=Session())
        actions = c.request_actions(b"x", {})
        assert c.last_error is None and len(actions) == 8 and len(actions[0]) == 10
        m = action_to_move(actions[0], tilt_deg=0.0, dt=1.0)
        assert m.vx > 0 and m.yaw_rate > 0      # forward, and LEFT
    finally:
        srv.shutdown()
        srv.server_close()
