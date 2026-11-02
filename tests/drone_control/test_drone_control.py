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


# --- the camera channel (second, independent link) -------------------------------
from src.drone_control.camera import CameraStream, Frame          # noqa: E402
from src.drone_control.sources import VisionPolicySource          # noqa: E402


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
    cam = CameraStream("x", open_capture=lambda u: FakeCapture(["a", "b", "c"]))
    cam.start()
    deadline = _t.time() + 2.0
    while _t.time() < deadline and cam.frames_read < 3:
        _t.sleep(0.01)
    cam.stop()
    assert cam.latest().image == "c"          # not "a" -- no backlog
    assert cam.frames_read >= 3


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


def test_vision_source_withholds_command_when_video_is_stale():
    """A dead camera must stop the drone, not feed the policy a photo of the past."""
    class Client:
        def request_command(self, *a, **k):
            raise AssertionError("must not call the policy without a fresh frame")
    src = VisionPolicySource(_FixedCam(None), Client())
    assert src.next_command(None, 0.0) is None
    assert src.skipped_stale_frame == 1


def test_vision_source_passes_frame_through_to_policy(monkeypatch):
    import src.drone_control.sources as S
    monkeypatch.setattr(S, "encode_jpeg", lambda img, q=80: b"jpegbytes")
    seen = {}

    class Client:
        def request_command(self, jpeg, state, goal):
            seen["jpeg"], seen["goal"] = jpeg, goal
            return VelocityCommand(vx=1.0, yaw_rate=5.0, note="ok")

    src = VisionPolicySource(_FixedCam(Frame(image="img", seq=42)), Client(),
                             goal={"brief": "orbit"})
    cmd = src.next_command(None, 0.0)
    assert seen["jpeg"] == b"jpegbytes" and seen["goal"] == {"brief": "orbit"}
    assert cmd.vx == 1.0 and cmd.seq == 42 and cmd.source == "vision-policy"


def test_vision_source_returns_none_when_policy_fails():
    class Dead:
        def request_command(self, *a, **k): return None
    import src.drone_control.sources as S
    orig = S.encode_jpeg
    S.encode_jpeg = lambda img, q=80: b"x"
    try:
        src = VisionPolicySource(_FixedCam(Frame(image="i", seq=1)), Dead())
        assert src.next_command(None, 0.0) is None
    finally:
        S.encode_jpeg = orig


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
