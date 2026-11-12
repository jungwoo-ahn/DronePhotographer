"""Recording a flight: the camera's stream copied to a file, and the flight log beside it."""

import json
import subprocess
import time

import pytest

from src.drone_control.flightlog import FlightLog
from src.drone_control.recorder import RecordingOpener, capture_command, find_ffmpeg
from src.drone_control.shots import load_shots, read_shot_list

FFMPEG = find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="no ffmpeg (pip install imageio-ffmpeg)")


def _events(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _wait(fn, timeout_s=20.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


# --- the recorder ------------------------------------------------------------------------
def _clip(tmp_path, seconds=2):
    """An H.265 clip like the camera's, to stand in for its stream."""
    clip = tmp_path / "camera.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=30", "-t", str(seconds), "-c:v", "libx265",
                    "-preset", "ultrafast", "-x265-params", "log-level=error", str(clip)],
                   check=True)
    return clip


def _count_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    return n


@needs_ffmpeg
def test_one_connection_gives_the_pilot_frames_and_records_the_stream(tmp_path):
    clip = _clip(tmp_path)
    log = FlightLog(tmp_path / "flight.jsonl")
    opener = RecordingOpener(tmp_path / "flight", size=(320, 240), log=log)
    cap = opener(str(clip))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    log.close()
    assert len(frames) == 60 and frames[0].shape == (240, 320, 3)
    assert _count_frames(tmp_path / "flight" / "video.mkv") == 60     # copied, every frame
    events = _events(tmp_path / "flight.jsonl")
    assert [e["kind"] for e in events][0] == "video_start"
    end = next(e for e in events if e["kind"] == "video_end")
    assert end["seconds"] == pytest.approx(2.0, abs=0.1) and end["code"] == 0


@needs_ffmpeg
def test_the_pilots_camera_reads_through_the_recording_and_finishes_the_file(tmp_path):
    from src.drone_control.camera import CameraStream
    clip = _clip(tmp_path, seconds=2)
    opener = RecordingOpener(tmp_path / "flight", size=(320, 240))
    cam = CameraStream(str(clip), open_capture=opener, settle_s=0.0, reconnect_delay_s=0.2).start()
    assert _wait(lambda: cam.frames_read >= 30)
    cam.stop()
    # a file ends where a camera would not, so each reconnection was a new part
    assert opener.parts[0] == "video.mkv" and _count_frames(tmp_path / "flight" / "video.mkv") == 60


def test_the_command_copies_the_stream_and_decodes_it_once():
    cmd = capture_command("ffmpeg", "rtsp://192.168.144.25:8554/main.264",
                          __import__("pathlib").Path("x/video.mkv"), (1920, 1080))
    assert cmd.count("-i") == 1                                          # one connection
    assert cmd.index("-progress") < cmd.index("-i")                       # global, before input
    assert cmd[cmd.index("-c") + 1] == "copy" and "matroska" in cmd
    assert cmd[-1] == "pipe:1" and "bgr24" in cmd and "1920x1080" in cmd
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"                # lost packets re-sent
    udp = capture_command("ffmpeg", "rtsp://cam/x", __import__("pathlib").Path("v.mkv"), (8, 8), "udp")
    assert udp[udp.index("-rtsp_transport") + 1] == "udp"
    assert "-rtsp_transport" not in capture_command("ffmpeg", "clip.mkv",
                                                    __import__("pathlib").Path("v.mkv"), (8, 8))


@needs_ffmpeg
def test_a_connection_that_got_no_picture_leaves_no_file_and_gives_its_name_back(tmp_path):
    opener = RecordingOpener(tmp_path / "flight", size=(320, 240))
    for _ in range(3):
        cap = opener(str(tmp_path / "no-such-camera.mkv"))
        assert cap.read() == (False, None)
        cap.release()
    assert list((tmp_path / "flight").glob("*.mkv")) == []
    assert opener.parts == ["video.mkv"] and opener.status()["parts"] == 0


@needs_ffmpeg
def test_the_parts_are_joined_into_one_mp4_that_can_be_sought(tmp_path):
    import shutil
    from src.drone_control.recorder import make_mp4
    flight = tmp_path / "flight"
    flight.mkdir()
    shutil.copy(_clip(tmp_path, seconds=2), flight / "video.mkv")
    (tmp_path / "camera.mkv").unlink()
    shutil.copy(_clip(tmp_path, seconds=3), flight / "video-2.mkv")
    (flight / "video-3.mkv").write_bytes(b"")                   # an empty one is skipped
    mp4 = make_mp4(flight)
    assert mp4 == flight / "flight.mp4"
    assert _count_frames(mp4) == 150                              # 2 s + 3 s at 30 fps
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(mp4)], capture_output=True,
                           text=True).stderr
    assert "hevc" in probe and "Duration: 00:00:05" in probe
    assert sorted(p.name for p in flight.iterdir()) == ["flight.mp4", "video-2.mkv",
                                                        "video-3.mkv", "video.mkv"]
    (tmp_path / "empty").mkdir()
    assert make_mp4(tmp_path / "empty") is None                # nothing recorded, nothing made


class _FakeProc:
    """An ffmpeg that sends nothing until it is told to stop or killed."""

    def __init__(self):
        import threading
        self.gone = threading.Event()
        self.got = b""
        self.stdin = self
        self.stdout = self
        self.stderr = iter(())
        self.killed = False

    def write(self, b):
        self.got += b
        if b"q" in self.got:
            self.gone.set()

    def flush(self):
        pass

    def readinto(self, view):
        self.gone.wait()
        return 0

    def read(self, n=-1):
        self.gone.wait()
        return b""

    def poll(self):
        return 0 if self.gone.is_set() else None

    def wait(self, timeout=None):
        if not self.gone.wait(timeout if timeout is not None else 60):
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return 0

    def kill(self):
        self.killed = True
        self.gone.set()


def test_a_camera_that_goes_quiet_is_dropped_so_the_pilot_reconnects():
    from src.drone_control.recorder import RecordingCapture
    proc = _FakeProc()
    cap = RecordingCapture(["ffmpeg"], (4, 2), name="video.mkv", popen=lambda *a, **k: proc,
                           stall_s=0.3)
    assert cap.read() == (False, None)                  # returned, rather than waiting forever
    assert proc.killed and "no video" in cap.error


def test_release_asks_ffmpeg_to_finish_the_file():
    from src.drone_control.recorder import RecordingCapture
    proc = _FakeProc()
    cap = RecordingCapture(["ffmpeg"], (4, 2), name="video.mkv", popen=lambda *a, **k: proc,
                           stall_s=30)
    cap.release()
    assert proc.got == b"q" and not proc.killed


@pytest.mark.skipif(__import__("os").name != "nt", reason="the job object is Windows only")
def test_ffmpeg_cannot_outlive_a_pilot_that_is_killed(tmp_path):
    """A pilot that dies without stopping (Task Manager, a crash) takes its ffmpeg with it."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    pid_file = tmp_path / "child.pid"
    code = (f"import sys, subprocess, os; sys.path.insert(0, {str(repo)!r});"
            "from src.drone_control.recorder import tie_to_this_process;"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            "assert tie_to_this_process(p);"
            f"open({str(pid_file)!r}, 'w').write(str(p.pid));"
            "os._exit(0)")                      # dies without cleaning up, like a kill
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
    pid = int(pid_file.read_text())

    def alive():
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                             text=True).stdout
        return str(pid) in out
    assert _wait(lambda: not alive(), timeout_s=10)


def test_without_ffmpeg_it_says_so(tmp_path, monkeypatch):
    from src.drone_control import recorder
    monkeypatch.setattr(recorder, "find_ffmpeg", lambda: None)
    opener = RecordingOpener(tmp_path)
    assert "ffmpeg" in opener.error and not opener.status()["recording"]


# --- the flight log -------------------------------------------------------------------------
def _snap(**kw):
    base = {"status": "flying", "rung": "idle", "reason": "", "shot": "eval-20",
            "telemetry": {"altitude": 5.0, "north": 0.0, "east": 0.0},
            "command": {"vx": 0.0}, "camera": {"tilt": -10.0, "target": -10.0},
            "links": {"video": {"age_ms": 30.0}},
            "model": {"answered_wall": None, "plan": [], "last_pass_s": None},
            "move": {"record": [], "running": None}}
    base.update(kw)
    return base


def test_the_log_writes_changes_once_and_the_state_twice_a_second(tmp_path):
    t = [100.0]
    log = FlightLog(tmp_path / "f.jsonl", clock=lambda: t[0])
    log.tick(_snap())
    t[0] += 0.1
    log.tick(_snap())                                   # nothing new, too soon for a state
    t[0] += 0.5
    plan = [{"n": 1, "x": 0.2, "text": "Move closer"}]
    log.tick(_snap(model={"answered_wall": 99.0, "plan": plan, "last_pass_s": 3.1},
                   move={"running": {"wall": 1.0, "text": "Move closer", "seq": 1, "note": "n",
                                     "command": {"vx": 0.1}},
                         "record": [{"wall": 2.0, "seq": 1, "outcome": "flown", "text": "Move closer"}]}))
    log.tick(_snap(rung="estop", reason="stop pressed", shot="eval-22"))
    log.close()
    kinds = [e["kind"] for e in _events(tmp_path / "f.jsonl")]
    assert kinds == ["shot", "state", "move", "run", "plan", "state", "shot", "stop"]
    e = _events(tmp_path / "f.jsonl")
    assert e[1]["camera"] == {"tilt": -10.0, "target": -10.0} and e[4]["plan"] == plan


def test_the_log_ignores_the_ground_checks(tmp_path):
    log = FlightLog(tmp_path / "f.jsonl")
    log.tick({"status": "starting", "phase": "Waiting"})
    log.close()
    assert (tmp_path / "f.jsonl").read_text() == ""


# --- the shot list ----------------------------------------------------------------------------
LIST = __import__("pathlib").Path(__file__).resolve().parents[2] / "configs" / "shot_lists" / "female_student.json"


def test_the_female_student_list_names_real_shots_in_its_order():
    listed = read_shot_list(LIST)
    shots = load_shots(only=listed)
    assert [s["id"] for s in shots] == [e["id"] for e in listed["shots"]]
    assert len(shots) == 10 and listed["start"] in [s["id"] for s in shots]
    assert shots[0]["name"].endswith(", standing")
    assert all(s["group"].startswith("From the ") for s in shots)


def test_a_list_naming_a_shot_that_does_not_exist_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"shots": [{"id": "eval-99"}]}))
    with pytest.raises(ValueError, match="eval-99"):
        load_shots(only=read_shot_list(bad))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"shots": []}))
    with pytest.raises(ValueError):
        read_shot_list(empty)


def test_with_a_list_and_no_goal_file_the_pilot_starts_from_the_lists_start():
    import importlib.util
    path = LIST.parents[2] / "scripts" / "run_sitl_pilot.py"
    spec = importlib.util.spec_from_file_location("run_sitl_pilot", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    a = type("A", (), {"goal_file": None, "shots": str(LIST)})()
    goal = pilot.starting_goal(a)
    start = next(s for s in load_shots(only=read_shot_list(LIST)) if s["id"] == "eval-20")
    assert goal == {"profile": start["profile"]} and a.start_shot == "eval-20"
    with pytest.raises(SystemExit):
        pilot.starting_goal(type("A", (), {"goal_file": None, "shots": None})())
