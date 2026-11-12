"""Picking a shot in the panel, the model's status, and the path a sequence leads along."""

import json
import math
import threading
import time
import urllib.request

import pytest

from src.drone_control import sources
from src.drone_control.approval import ApprovalGate
from src.drone_control.camera import Frame
from src.drone_control.safety import VehicleState
from src.drone_control.shots import FRAME_H, FRAME_W, PROFILE_KEYS, load_shots, sentence
from src.drone_control.sources import ActionSequenceSource, plan_path
from src.drone_control.ui import UiServer, UiState

EXAMPLE = {"occupancy": 16.0, "body_in_frame_ratio": 100.0, "subject_bearing_deg": 137.0,
           "cam_to_obj_elevation_deg": -1.0, "object_center_x": 170.0, "object_center_y": 567.0,
           "bbox_x_offset": 163.0, "bbox_y_offset": 191.0, "top_cut_frac": 0.0,
           "bot_cut_frac": 0.0, "visible_frac": 1.0}
AHEAD = [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def _turn(deg):
    """An action that only turns the camera left by `deg` (OpenCV frame: about its down axis)."""
    t = -math.radians(deg)
    return [0.0, 0.0, 0.0, math.cos(t), 0.0, -math.sin(t), 0.0, 1.0, 0.0, 0.0]


def _tilt(deg):
    """An action that only tilts the camera up by `deg`."""
    t = math.radians(deg)
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, math.cos(t), math.sin(t), 0.0]


# --- the shots -----------------------------------------------------------------------
def test_the_list_starts_with_the_goal_file_then_eight_from_each_side():
    shots = load_shots({"profile": EXAMPLE})
    assert shots[0]["id"] == "start" and shots[0]["profile"] == EXAMPLE
    rest = shots[1:]
    assert len(rest) == 64
    groups = [s["group"] for s in rest]
    assert {g: groups.count(g) for g in groups} == {g: 8 for g in groups}
    assert groups[0] == "From the front"
    for s in rest:
        assert all(k in s["profile"] for k in PROFILE_KEYS)
    assert len({s["id"] for s in shots}) == len(shots)


def test_a_shot_is_described_as_the_model_will_read_it():
    text = sentence(EXAMPLE)
    assert text.startswith("Move the camera to achieve this shot: a wide shot of the subject "
                           "from the subject's back-right")
    assert "left third and lower in the frame" in text and "occupancy 16%" in text


def test_each_shot_names_its_goal_photo_when_it_has_been_fetched(tmp_path):
    for name in ("eval-00.jpg", "eval-05.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8jpeg")
    shots = {s["id"]: s for s in load_shots({"profile": EXAMPLE}, previews=tmp_path)}
    assert shots["eval-05"]["image"] == "eval-05.jpg"
    assert shots["eval-01"]["image"] is None                  # not fetched
    # the example goal file is episode 0's goal, so it shows that episode's photo
    assert shots["start"]["image"] == "eval-00.jpg"
    assert all(s["image"] is None for s in load_shots({"profile": EXAMPLE}))   # no previews


def test_a_goal_file_of_its_own_has_no_photo(tmp_path):
    (tmp_path / "eval-00.jpg").write_bytes(b"\xff\xd8jpeg")
    mine = dict(EXAMPLE, occupancy=30.0)
    assert load_shots({"profile": mine}, previews=tmp_path)[0]["image"] is None


def test_the_fetch_script_shrinks_a_render_to_the_panels_size():
    import importlib.util
    from pathlib import Path

    import cv2
    import numpy as np
    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_shot_previews.py"
    spec = importlib.util.spec_from_file_location("fetch_shot_previews", path)
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    ok, big = cv2.imencode(".jpg", np.full((768, 1024, 3), 90, np.uint8))
    small = cv2.imdecode(np.frombuffer(fetch.shrink(big.tobytes(), 512), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape == (384, 512, 3)
    with pytest.raises(ValueError):
        fetch.shrink(b"not a jpeg", 512)


def test_the_goal_files_shot_is_left_out_when_it_is_incomplete():
    assert all(s["id"] != "start" for s in load_shots({"profile": {"occupancy": 16}}))


def test_the_target_box_is_inside_the_models_view_for_the_example():
    left = EXAMPLE["object_center_x"] - EXAMPLE["bbox_x_offset"]
    bottom = EXAMPLE["object_center_y"] + EXAMPLE["bbox_y_offset"]
    assert 0 <= left and bottom <= FRAME_H and FRAME_W == 1024


# --- the path --------------------------------------------------------------------------
def test_the_path_goes_forward_for_a_forward_action():
    steps = plan_path([AHEAD, AHEAD], 0.0, 1.5)
    assert [round(s["x"], 6) for s in steps] == [0.3, 0.6]
    assert all(abs(s["y"]) < 1e-9 and abs(s["z"]) < 1e-9 for s in steps)
    assert steps[0]["n"] == 1 and steps[0]["text"] == "Move closer"


def test_after_a_turn_forward_is_the_new_forward():
    steps = plan_path([_turn(90), AHEAD], 0.0, 1.0)
    assert steps[0]["heading_deg"] == pytest.approx(90.0)
    assert steps[1]["x"] == pytest.approx(0.0, abs=1e-9) and steps[1]["y"] == pytest.approx(0.3)


def test_each_step_starts_from_the_tilt_the_last_one_left():
    steps = plan_path([_tilt(-30), AHEAD], 0.0, 1.0)
    assert steps[0]["tilt"] == pytest.approx(-30.0)
    # forward with the camera 30° down is forward and down
    assert steps[1]["x"] == pytest.approx(0.3 * math.cos(math.radians(30)))
    assert steps[1]["z"] == pytest.approx(-0.3 * math.sin(math.radians(30)))


def test_without_a_gimbal_the_tilt_stays_put():
    steps = plan_path([_tilt(-30), AHEAD], 0.0, 1.0, follow_tilt=False)
    assert steps[1]["z"] == pytest.approx(0.0, abs=1e-9)


# --- a new shot ---------------------------------------------------------------------------
class _Cam:
    def fresh_frame(self, now=None):
        return Frame(image="img", seq=1)


class _Client:
    def __init__(self, gate_event=None):
        self.asked, self.last_error = [], None
        self.release = gate_event

    def request_actions(self, jpeg, goal, timeout_s=None):
        self.asked.append(goal)
        if self.release is not None:
            self.release.wait(5.0)
        return [list(AHEAD) for _ in range(3)]


def _source(monkeypatch, client):
    monkeypatch.setattr(sources, "encode_jpeg", lambda img, q: b"jpeg")
    return ActionSequenceSource(_Cam(), client, {"profile": {"occupancy": 1}}, move_s=1.5)


def _wait(fn, tries=300):
    for _ in range(tries):
        out = fn()
        if out:
            return out
        time.sleep(0.01)
    return None


def test_the_status_says_thinking_then_shows_the_plan(monkeypatch):
    release = threading.Event()
    src = _source(monkeypatch, _Client(release))
    assert src.next_command(None, 0.0) is None
    s = src.status()
    assert s["asking"] and s["asking_for_s"] is not None and s["plan"] == []
    release.set()
    cmd = _wait(lambda: src.next_command(None, 0.0))
    s = src.status()
    assert not s["asking"] and len(s["plan"]) == 3 and s["offered"] == 1
    assert s["last_pass_s"] is not None and cmd.seq == 1


def test_a_new_shot_drops_the_old_sequence_and_asks_with_the_new_goal(monkeypatch):
    client = _Client()
    src = _source(monkeypatch, client)
    _wait(lambda: src.next_command(None, 0.0))
    src.set_goal({"profile": {"occupancy": 2}})
    assert src.status()["plan"] == []
    cmd = _wait(lambda: src.next_command(None, 0.0))
    assert cmd.seq == 1 and client.asked[-1] == {"profile": {"occupancy": 2}}


def test_an_answer_for_the_old_shot_is_thrown_away(monkeypatch):
    release = threading.Event()
    client = _Client(release)
    src = _source(monkeypatch, client)
    assert src.next_command(None, 0.0) is None          # asking with the old goal
    _wait(lambda: client.asked)
    src.set_goal({"profile": {"occupancy": 2}})
    release.set()
    cmd = _wait(lambda: src.next_command(None, 0.0))
    assert cmd is not None and client.asked[-1] == {"profile": {"occupancy": 2}}
    assert len(client.asked) == 2


def test_a_move_for_the_old_shot_is_never_put_on_offer(monkeypatch):
    src = _source(monkeypatch, _Client())
    gate = ApprovalGate(src, offer_s=float("inf"))
    real = src.next_command

    def racing(state, now):
        cmd = real(state, now)
        if cmd is not None:
            src.set_goal({"profile": {"occupancy": 2}})     # picked just as it was worked out
        return cmd
    src.next_command = racing
    state = VehicleState(altitude=5.0)
    for _ in range(100):
        gate.next_command(state, 0.0)
        time.sleep(0.01)
        if src.forward_passes >= 2:
            break
    assert gate.snapshot(0.0)["offer"] is None


# --- the panel ------------------------------------------------------------------------------
def _strict(body: bytes):
    """Parse as a browser does: `Infinity` and `NaN` are not JSON."""
    def refuse(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(body, parse_constant=refuse)


def test_the_state_is_json_a_browser_can_read_when_an_offer_never_expires(monkeypatch):
    src = _source(monkeypatch, _Client())
    gate = ApprovalGate(src, offer_s=float("inf"))
    state = VehicleState(altitude=5.0)
    _wait(lambda: (gate.next_command(state, 0.0), gate.snapshot(0.0)["offer"])[1])
    ui = UiState()
    ui._snap = {"move": gate.snapshot(0.0), "odd": float("nan")}

    class Sup:
        stats = {}

    server = UiServer(ui, Sup(), port=0, gate=gate).start()
    try:
        url = f"http://127.0.0.1:{server._srv.server_address[1]}/api/state"
        snap = _strict(urllib.request.urlopen(url).read())
        assert snap["move"]["offer_s"] is None and snap["move"]["offer"]["expires_in_s"] is None
        assert snap["odd"] is None
    finally:
        server.stop()



def test_the_panel_lists_the_shots_and_a_pick_takes_back_the_offer(monkeypatch):
    src = _source(monkeypatch, _Client())
    gate = ApprovalGate(src, offer_s=float("inf"))
    ui = UiState()
    shots = load_shots({"profile": EXAMPLE})
    ui.attach_model(src, shots)

    class Sup:
        stats = {}

    server = UiServer(ui, Sup(), port=0, gate=gate).start()
    base = f"http://127.0.0.1:{server._srv.server_address[1]}"
    try:
        state = VehicleState(altitude=5.0)
        _wait(lambda: (gate.next_command(state, 0.0), gate.snapshot(0.0)["offer"])[1])
        listed = json.loads(urllib.request.urlopen(base + "/api/shots").read())
        assert [s["id"] for s in listed] == [s["id"] for s in shots]

        req = urllib.request.Request(base + "/api/shot?id=" + shots[5]["id"], method="POST")
        assert json.loads(urllib.request.urlopen(req).read()) == {"chosen": True}
        snap = gate.snapshot(0.0)
        assert snap["offer"] is None and snap["record"][0]["outcome"] == "withdrawn: new shot"
        assert src.goal == {"profile": shots[5]["profile"]}

        bad = urllib.request.Request(base + "/api/shot?id=nope", method="POST")
        assert json.loads(urllib.request.urlopen(bad).read()) == {"chosen": False}
    finally:
        server.stop()


def test_the_panel_serves_a_shots_photo_and_nothing_else(monkeypatch, tmp_path):
    import urllib.error
    (tmp_path / "eval-00.jpg").write_bytes(b"\xff\xd8photo")
    (tmp_path / "secret.txt").write_text("no")
    ui = UiState()
    ui.attach_model(_source(monkeypatch, _Client()), load_shots({"profile": EXAMPLE}, previews=tmp_path),
                    previews=tmp_path)

    class Sup:
        stats = {}

    server = UiServer(ui, Sup(), port=0).start()
    base = f"http://127.0.0.1:{server._srv.server_address[1]}"
    try:
        assert urllib.request.urlopen(base + "/api/shot.jpg?id=start").read() == b"\xff\xd8photo"
        for bad in ("eval-01", "secret.txt", "..%2Fsecret.txt", ""):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + "/api/shot.jpg?id=" + bad)
            assert e.value.code == 404
    finally:
        server.stop()
