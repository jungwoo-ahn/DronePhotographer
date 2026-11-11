"""The SIYI driver, against the manual's own frames and a fake gimbal on a real UDP socket."""

import math
import socket
import struct
import threading
import time

import pytest

from src.drone_control import sources
from src.drone_control.approval import ApprovalGate

from src.drone_control.gimbal import (
    CMD_CENTER,
    CMD_GET_ATTITUDE,
    CMD_GET_ENCODING,
    CMD_REBOOT,
    CMD_SET_ANGLE,
    CMD_SET_ENCODING,
    ENCODING,
    SiyiGimbal,
    TiltHold,
    crc16,
    decode,
    encode,
)
from src.drone_control.sources import ActionSequenceSource


@pytest.mark.parametrize("hex_frame", [
    "55 66 01 02 00 00 00 07 64 64 3d cf",      # rotate 100 100
    "55 66 01 01 00 00 00 08 01 d1 12",         # centre
    "55 66 01 00 00 00 00 0d e8 05",            # acquire attitude
    "55 66 01 01 00 00 00 0c 00 34 ce",         # take picture
    "55 66 01 00 00 00 00 01 64 c4",            # firmware version
    "55 66 01 01 00 00 00 04 01 bc 57",         # auto focus
])
def test_frames_match_the_manual_byte_for_byte(hex_frame):
    frame = bytes.fromhex(hex_frame)
    cmd, data = decode(frame)
    assert encode(cmd, data) == frame


def test_decode_refuses_anything_damaged():
    good = encode(CMD_GET_ATTITUDE, b"\x01\x02")
    assert decode(good) == (CMD_GET_ATTITUDE, b"\x01\x02")
    assert decode(good[:-1]) is None                        # truncated
    assert decode(good[:-1] + bytes([good[-1] ^ 1])) is None    # bad checksum
    assert decode(b"\x00" + good[1:]) is None               # bad start
    assert decode(b"") is None


def test_crc_is_over_the_whole_frame_before_it():
    frame = encode(CMD_CENTER, b"\x01")
    assert struct.unpack("<H", frame[-2:])[0] == crc16(frame[:-2])


class FakeGimbal:
    """Answers like the manual says: angles in 0.1° and yaw, pitch, roll in that order."""

    def __init__(self, noise_first=False, damaged=False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.pitch = -123                       # -12.3°
        self.stream = (1280, 720, 1607)
        self.encoding_ok = 1
        self.got = []
        self.noise_first, self.damaged = noise_first, damaged
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                frame, addr = self.sock.recvfrom(1024)
            except OSError:
                continue
            cmd, data = decode(frame)
            self.got.append((cmd, data))
            if self.noise_first:
                self.sock.sendto(encode(0x01, b"\x00" * 12), addr)
            attitude = struct.pack("<hhh", 0, self.pitch, 0)
            if cmd == CMD_SET_ANGLE:
                self.pitch = struct.unpack("<hh", data)[1]
                reply = encode(cmd, attitude)
            elif cmd == CMD_GET_ATTITUDE:
                reply = encode(cmd, attitude + b"\x00" * 6)
            elif cmd == CMD_GET_ENCODING:
                reply = encode(cmd, ENCODING.pack(data[0], 2, *self.stream) + b"\x1e")
            elif cmd == CMD_SET_ENCODING:
                self.stream = struct.unpack_from("<HHH", data, 2)
                reply = encode(cmd, bytes([data[0], self.encoding_ok]))
            else:
                reply = encode(cmd, b"\x01")
            self.sock.sendto(reply[:-1] + b"\x00" if self.damaged else reply, addr)

    def close(self):
        self.stop.set()
        self.thread.join(1.0)
        self.sock.close()


@pytest.fixture
def fake():
    f = FakeGimbal()
    yield f
    f.close()


def _driver(f):
    return SiyiGimbal("127.0.0.1", f.port, timeout_s=0.3)


def test_reads_the_tilt_in_degrees(fake):
    assert _driver(fake).tilt() == -12.3


def test_set_tilt_sends_tenths_of_a_degree_and_reports_the_current_angle(fake):
    g = _driver(fake)
    assert g.set_tilt(-30.0) == -12.3           # where it was when it answered
    assert fake.got[-1] == (CMD_SET_ANGLE, struct.pack("<hh", 0, -300))
    assert g.tilt() == -30.0


@pytest.mark.parametrize("asked,sent", [(40.0, 250), (-120.0, -900), (60.5 - 50, 105)])
def test_set_tilt_stays_inside_the_gimbals_range(fake, asked, sent):
    _driver(fake).set_tilt(asked)
    assert struct.unpack("<hh", fake.got[-1][1])[1] == sent


def test_center(fake):
    assert _driver(fake).center() is True
    assert fake.got[-1] == (CMD_CENTER, b"\x01")


def test_a_silent_gimbal_answers_none_rather_than_raising():
    silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    silent.bind(("127.0.0.1", 0))
    g = SiyiGimbal("127.0.0.1", silent.getsockname()[1], timeout_s=0.05)
    assert g.tilt() is None and g.set_tilt(-10.0) is None and g.center() is False
    silent.close()


def test_steady_tilt_needs_the_readings_to_agree(fake):
    g = _driver(fake)
    assert g.steady_tilt(samples=3, gap_s=0.0) == pytest.approx(-12.3)
    readings = iter([-94.0, 149.4, -94.0])
    g.tilt = lambda: next(readings)
    assert g.steady_tilt(samples=3, gap_s=0.0) is None


def test_steady_tilt_needs_every_reading():
    g = SiyiGimbal("127.0.0.1", 9)
    readings = iter([-10.0, None, -10.0])
    g.tilt = lambda: next(readings)
    assert g.steady_tilt(samples=3, gap_s=0.0) is None
    g.close()


def test_reads_the_real_cameras_answer_for_its_main_stream():
    assert ENCODING.unpack_from(bytes.fromhex("010200 05d0 02 4706 1e".replace(" ", ""))) == (
        1, 2, 1280, 720, 1607)


def test_main_stream_can_be_read_and_set(fake):
    g = _driver(fake)
    assert g.main_stream() == {"width": 1280, "height": 720, "kbps": 1607, "fps": 30}
    assert g.set_main_stream(1920, 1080, 4096) is True
    assert fake.got[-1] == (CMD_SET_ENCODING, struct.pack("<BBHHHB", 1, 2, 1920, 1080, 4096, 0))
    assert g.main_stream()["width"] == 1920


def test_reboot_restarts_the_camera_and_leaves_the_gimbal_alone(fake):
    assert _driver(fake).reboot_camera() is True
    assert fake.got[-1] == (CMD_REBOOT, b"\x01\x00")


def test_the_pilot_restarts_the_camera_only_when_it_changed_the_stream(fake, capsys):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_sitl_pilot.py"
    spec = importlib.util.spec_from_file_location("run_sitl_pilot", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    g = _driver(fake)
    assert pilot.use_full_hd(g, 4096) is True
    assert [cmd for cmd, _ in fake.got[-2:]] == [CMD_SET_ENCODING, CMD_REBOOT]
    assert pilot.use_full_hd(g, 4096) is False          # already 1080p: left alone
    assert fake.got[-1][0] == CMD_GET_ENCODING
    fake.stream, fake.encoding_ok = (1280, 720, 1607), 0
    assert pilot.use_full_hd(g, 4096) is False          # refused: no restart
    assert CMD_REBOOT not in [cmd for cmd, _ in fake.got[-2:]]


def test_a_refused_stream_setting_says_so(fake):
    fake.encoding_ok = 0
    assert _driver(fake).set_main_stream(1920, 1080, 4096) is False


def test_a_silent_camera_has_no_stream_settings():
    silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    silent.bind(("127.0.0.1", 0))
    g = SiyiGimbal("127.0.0.1", silent.getsockname()[1], timeout_s=0.05)
    assert g.main_stream() is None and g.set_main_stream(1920, 1080, 4096) is False
    silent.close()


def test_replies_to_something_else_are_ignored():
    f = FakeGimbal(noise_first=True)
    assert _driver(f).tilt() == -12.3
    f.close()


def test_a_damaged_reply_is_no_reply():
    f = FakeGimbal(damaged=True)
    g = SiyiGimbal("127.0.0.1", f.port, timeout_s=0.1)
    assert g.tilt() is None
    f.close()


class SimulatedGimbal:
    """Reports its pitch from the horizon. `follows_airframe` is what the bench seemed to show."""

    def __init__(self, follows_airframe=True):
        self.follows_airframe = follows_airframe
        self.airframe = 0.0
        self.commanded = 0.0
        self.sent = []
        self.next_readings = []

    def tilt(self):
        if self.next_readings:
            return self.next_readings.pop(0)
        return self.commanded + (self.airframe if self.follows_airframe else 0.0)

    def set_tilt(self, degrees):
        self.commanded = degrees
        self.sent.append(degrees)


def _run_hold(hold, steps=30):
    for _ in range(steps):
        hold.step()


@pytest.mark.parametrize("follows", [True, False])
def test_the_hold_keeps_the_angle_when_the_aircraft_tips(follows):
    g = SimulatedGimbal(follows)
    hold = TiltHold(g, -45.0)
    g.set_tilt(-45.0)
    _run_hold(hold)
    g.airframe = 20.0
    _run_hold(hold)
    assert g.tilt() == pytest.approx(-45.0, abs=1.0)
    assert hold.settled()


def test_the_hold_goes_to_a_new_target_and_says_when_it_has():
    g = SimulatedGimbal()
    hold = TiltHold(g, 0.0)
    _run_hold(hold, 3)
    hold.set_target(-30.0)
    assert not hold.settled()
    _run_hold(hold)
    assert hold.settled() and hold.tilt == pytest.approx(-30.0, abs=1.0)


def test_one_wild_reading_moves_nothing():
    g = SimulatedGimbal()
    hold = TiltHold(g, -45.0)
    g.set_tilt(-45.0)
    _run_hold(hold, 5)
    sent = len(g.sent)
    g.next_readings = [-40.0 - 100.0, 149.4, -94.0 + 60.0, -170.0]
    _run_hold(hold, 4)
    assert len(g.sent) == sent and hold.tilt is None


def test_a_silent_or_upside_down_gimbal_is_left_alone():
    g = SimulatedGimbal()
    hold = TiltHold(g, -45.0)
    g.next_readings = [None] * 4 + [179.9] * 4
    _run_hold(hold, 8)
    assert g.sent == [] and hold.tilt is None


def test_the_hold_never_asks_for_more_than_the_gimbal_has():
    g = SimulatedGimbal()
    hold = TiltHold(g, -85.0)
    g.airframe = 40.0           # it cannot reach -85 from here
    _run_hold(hold, 60)
    assert min(g.sent) >= -90.0 and max(g.sent) <= 25.0


def test_the_target_stays_inside_the_range():
    hold = TiltHold(SimulatedGimbal(), 80.0)
    assert hold.target == 25.0
    hold.set_target(-200.0)
    assert hold.target == -90.0


def _action_tilting_by(degrees):
    """A pure rotation in the camera frame (OpenCV: right, down, forward), up positive."""
    e = math.radians(degrees)
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.cos(e), math.sin(e), 0.0]


class _Frames:
    class _F:
        image = None

    def fresh_frame(self):
        return self._F()


class _Client:
    calls = 0

    def request_actions(self, jpeg, goal, timeout_s=None):
        self.calls += 1
        return [_action_tilting_by(-10.0), _action_tilting_by(-10.0)]


def _sequence(monkeypatch, hold):
    monkeypatch.setattr(sources, "encode_jpeg", lambda image, quality: b"x")
    client = _Client()
    return ActionSequenceSource(_Frames(), client, {}, move_s=1.0, hold=hold), client


def _ask(source):
    for _ in range(200):
        cmd = source.next_command(None, 0.0)
        if cmd is not None:
            return cmd
        time.sleep(0.005)
    return None


def test_only_an_approved_action_aims_the_camera(monkeypatch):
    g = SimulatedGimbal()
    hold = TiltHold(g, 0.0)
    _run_hold(hold, 3)
    source, _ = _sequence(monkeypatch, hold)
    gate = ApprovalGate(source, hold_s=1.0, offer_s=float("inf"))
    for _ in range(200):
        gate.next_command(None, 0.0)
        if gate.snapshot(0.0)["offer"] is not None:
            break
        time.sleep(0.005)
    assert hold.target == 0.0
    assert "The camera goes to -10" in gate.snapshot(0.0)["offer"]["note"]
    assert gate.accept(0.0)
    assert hold.target == pytest.approx(-10.0, abs=0.01)


def test_an_offered_action_cannot_run_while_the_camera_is_off_target(monkeypatch):
    g = SimulatedGimbal()
    hold = TiltHold(g, 0.0)
    _run_hold(hold, 3)
    source, _ = _sequence(monkeypatch, hold)
    gate = ApprovalGate(source, hold_s=1.0, offer_s=float("inf"))
    for _ in range(200):
        gate.next_command(None, 0.0)
        if gate.snapshot(0.0)["offer"] is not None:
            break
        time.sleep(0.005)
    g.next_readings = [179.9, 179.9]
    _run_hold(hold, 2)
    assert gate.accept(0.0) is False
    assert gate.snapshot(0.0)["offer"] is not None      # still there for when it comes back
    _run_hold(hold)
    assert gate.accept(0.0) is True


def test_nothing_is_asked_or_offered_until_the_camera_has_arrived(monkeypatch):
    hold = TiltHold(SimulatedGimbal(), -40.0)       # nowhere near yet, and never stepped
    source, client = _sequence(monkeypatch, hold)
    assert source.next_command(None, 0.0) is None
    assert client.calls == 0
    _run_hold(hold)
    assert _ask(source) is not None
    assert client.calls == 1


def test_the_next_action_is_worked_out_from_the_held_angle(monkeypatch):
    hold = TiltHold(SimulatedGimbal(), -30.0)
    _run_hold(hold)
    source, _ = _sequence(monkeypatch, hold)
    assert "to -40" in _ask(source).note


def test_a_bumped_camera_stops_the_offers_until_it_is_back(monkeypatch):
    g = SimulatedGimbal()
    hold = TiltHold(g, -30.0)
    _run_hold(hold)
    source, _ = _sequence(monkeypatch, hold)
    assert _ask(source) is not None
    g.next_readings = [179.9, 179.9]
    _run_hold(hold, 2)
    assert source.next_command(None, 0.0) is None
    _run_hold(hold)
    assert source.next_command(None, 0.0) is not None
