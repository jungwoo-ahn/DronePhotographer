"""Astra-style pilot: action-vector signs, velocity integration, context accumulation.

The signs matter more than usual here. Astra's yaw_rate is LEFT-positive, which happens to
match this repr's `dyaw` -- unlike every other prompted baseline in this repo, which needed
`yaw_from_right_positive`. These tests pin that so nobody "fixes" it into a bug.
"""

import json
import math

import numpy as np
import pytest

from src.policy.astra.pilot import (DT, FIELDS, MAX_VX, MAX_VY, MAX_VZ, MAX_YAW_RATE,
                                    AstraPilot, apply_velocity, clamp_command)
from src.utils.rotation_utils import make_camera_basis_from_forward_up

POS = np.array([0.0, 0.0, 2.0], np.float32)
FWD = np.array([1.0, 0.0, 0.0], np.float32)
UP = np.array([0.0, 0.0, 1.0], np.float32)


def _cmd(**kw):
    return clamp_command({"status": "Searching", **kw})


def _axes():
    b = make_camera_basis_from_forward_up(FWD, UP)
    return b[:, 0], b[:, 1]          # right, up


# ---------------------------------------------------------------- published limits
def test_limits_match_the_blog():
    assert (MAX_VX, MAX_VY, MAX_VZ, MAX_YAW_RATE) == (5.0, 1.5, 1.0, 120.0)
    assert DT == pytest.approx(0.5)          # 2 Hz velocity mode


def test_clamp_saturates_and_survives_junk():
    c = clamp_command({"vx": 99, "vy": -99, "vz": 99, "yaw_rate_deg_s": 999,
                       "pitch_rate_deg_s": -999, "status": "Complete", "scratchpad": "n"})
    assert c["vx"] == MAX_VX and c["vy"] == -MAX_VY and c["vz"] == MAX_VZ
    assert c["yaw_rate_deg_s"] == MAX_YAW_RATE and c["status"] == "Complete"
    junk = clamp_command({"vx": "fast", "vy": None, "status": "whatever"})
    assert junk["vx"] == 0.0 and junk["vy"] == 0.0
    assert junk["status"] == "Searching"          # anything but Complete keeps flying


# ---------------------------------------------------------------- velocity integration
def test_vx_flies_along_the_heading_for_dt_seconds():
    p, _, _ = apply_velocity(POS, FWD, UP, _cmd(vx=2.0), dt=0.5)
    assert np.allclose(p, POS + np.array([1.0, 0.0, 0.0]), atol=1e-6)   # 2 m/s * 0.5 s


def test_vy_is_left_positive_and_vz_is_world_up():
    left = np.cross([0, 0, 1.0], FWD)                                   # +y for this heading
    p, _, _ = apply_velocity(POS, FWD, UP, _cmd(vy=1.0), dt=1.0)
    assert np.allclose(p - POS, left, atol=1e-6), "vy must push LEFT"
    p, _, _ = apply_velocity(POS, FWD, UP, _cmd(vz=1.0), dt=1.0)
    assert np.allclose(p - POS, [0, 0, 1.0], atol=1e-6)


def test_vx_is_levelled_to_horizontal_even_when_the_camera_is_pitched():
    """The blog levels vx; a camera aimed down must still fly horizontally, not into the floor."""
    fwd = np.array([1.0, 0.0, -1.0], np.float32) / math.sqrt(2)
    up = np.array([1.0, 0.0, 1.0], np.float32) / math.sqrt(2)
    p, _, _ = apply_velocity(POS, fwd, up, _cmd(vx=2.0), dt=1.0)
    assert p[2] == pytest.approx(POS[2], abs=1e-6), "vx leaked into altitude"
    assert p[0] == pytest.approx(POS[0] + 2.0, abs=1e-6)


def test_yaw_rate_is_left_positive_and_needs_no_sign_flip():
    """Astra's yaw is left-positive and so is this repr's dyaw -- unlike every other baseline."""
    right, _ = _axes()
    _, f, _ = apply_velocity(POS, FWD, UP, _cmd(yaw_rate_deg_s=60.0), dt=0.5)
    assert float(np.dot(f, right)) < -0.1, "positive yaw_rate must turn LEFT"
    _, f, _ = apply_velocity(POS, FWD, UP, _cmd(yaw_rate_deg_s=-60.0), dt=0.5)
    assert float(np.dot(f, right)) > 0.1


def test_pitch_rate_is_up_positive():
    _, up_axis = _axes()
    _, f, _ = apply_velocity(POS, FWD, UP, _cmd(pitch_rate_deg_s=40.0), dt=0.5)
    assert float(np.dot(f, up_axis)) > 0.1


def test_a_zero_command_is_a_hover():
    p, f, u = apply_velocity(POS, FWD, UP, _cmd(), dt=0.5)
    assert np.allclose(p, POS, atol=1e-6) and np.allclose(f, FWD, atol=1e-6)
    assert np.allclose(u, UP, atol=1e-6)


# ---------------------------------------------------------------- the pilot loop
class _VLM:
    """Records what it was shown so the context-accumulation claims can be checked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def ask_json(self, system, content):
        self.seen.append(content)
        return self.replies.pop(0) if self.replies else {}


def _goal_text():
    return {"mode": "text", "text": "a wide shot from the subject's back-right"}


def test_context_accumulates_frames_commands_and_scratchpads():
    vlm = _VLM([{"vx": 1.0, "status": "Searching", "scratchpad": "subject on my left"},
                {"vy": 1.0, "status": "Searching", "scratchpad": "circling left now"}])
    p = AstraPilot(vlm)
    p.act("/tmp/f0.jpg", _goal_text())
    p.act("/tmp/f1.jpg", _goal_text())

    first, second = vlm.seen
    imgs = lambda c: [b["path"] for b in c if b["type"] == "image"]   # noqa: E731
    text = lambda c: " ".join(b["text"] for b in c if b["type"] == "text")  # noqa: E731
    assert imgs(first) == ["/tmp/f0.jpg"]
    assert imgs(second) == ["/tmp/f0.jpg", "/tmp/f1.jpg"], "earlier frame was not resent"
    assert "subject on my left" in text(second), "earlier scratchpad was not carried forward"
    assert '"vx": 1.0' in text(second), "earlier command was not resent"
    assert p.scratchpad == "circling left now"


def test_context_window_caps_resent_frames_but_keeps_the_latest_note():
    vlm = _VLM([{"vx": 0.5, "status": "Searching", "scratchpad": f"n{i}"} for i in range(8)])
    p = AstraPilot(vlm, max_context_frames=3)
    for i in range(8):
        p.act(f"/tmp/f{i}.jpg", _goal_text())
    last = vlm.seen[-1]
    imgs = [b["path"] for b in last if b["type"] == "image"]
    assert len(imgs) == 4, "should resend the window plus the current frame"
    assert imgs[-1] == "/tmp/f7.jpg"
    assert "earlier steps omitted" in " ".join(b["text"] for b in last if b["type"] == "text")


def test_complete_is_the_termination_signal():
    vlm = _VLM([{"vx": 1.0, "status": "Searching"}, {"status": "Complete"}])
    p = AstraPilot(vlm)
    _, done = p.act("/tmp/a.jpg", _goal_text())
    assert done is False
    _, done = p.act("/tmp/b.jpg", _goal_text())
    assert done is True


def test_image_goal_mode_sends_the_reference_frame_first():
    vlm = _VLM([{"vx": 1.0, "status": "Searching"}])
    p = AstraPilot(vlm, goal_mode="image")
    p.act("/tmp/now.jpg", {"mode": "image", "image": "/tmp/goal.jpg"})
    imgs = [b["path"] for b in vlm.seen[0] if b["type"] == "image"]
    assert imgs == ["/tmp/goal.jpg", "/tmp/now.jpg"]


def test_a_dropped_reply_hovers_instead_of_crashing():
    class _Dead:
        def ask_json(self, system, content):
            return None

    p = AstraPilot(_Dead())
    cmd, done = p.act("/tmp/a.jpg", _goal_text())
    assert all(cmd[k] == 0.0 for k in FIELDS) and done is False
    assert p.failures == 1
    pos, fwd, up = apply_velocity(POS, FWD, UP, cmd)
    assert np.allclose(pos, POS) and np.allclose(fwd, FWD)


# ---------------------------------------------------------------- the harness loop
def test_fly_loop_steps_the_pose_renders_each_step_and_stops_on_complete(monkeypatch):
    """End-to-end through AstraPolicy.fly with a fake renderer -- no GPU, no gateway."""
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "clb", "scripts/closed_loop_eval_baselines.py")
    clb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clb)

    replies = [{"vx": 1.0, "status": "Searching", "scratchpad": "closing in"},
               {"vy": 1.0, "yaw_rate_deg_s": -20.0, "status": "Searching", "scratchpad": "arcing"},
               {"status": "Complete", "scratchpad": "framed"}]

    class _StubVLM:
        def __init__(self, *a, **k):
            self.calls = self.failures = 0

        def ask_json(self, system, content):
            self.calls += 1
            return replies[min(self.calls - 1, len(replies) - 1)]

    monkeypatch.setattr(clb, "GatewayVLM", _StubVLM)

    rendered = []

    class _Rend:
        def render(self, t, pos, fwd, up, want_profile=False):
            rendered.append((t, np.asarray(pos, float).copy()))
            return f"/tmp/astra_{t}.jpg"

    goal_frame = SimpleNamespace(
        raw={"subject_bearing_deg": 90.0, "occupancy": 20},
        camera_position=[4.0, 0.0, 2.0], camera_forward=[-1.0, 0.0, 0.0],
        camera_up=[0.0, 0.0, 1.0], subject_center=[0.0, 0.0, 1.0], subject_height=1.8)
    ep = SimpleNamespace(
        key="k", name="n", scene="s", goal_frame=goal_frame,
        start=SimpleNamespace(camera_position=[6.0, 0.0, 2.0],
                              camera_forward=[-1.0, 0.0, 0.0], camera_up=[0.0, 0.0, 1.0]))

    pol = clb.AstraPolicy("gpt-6-astra", goal_mode="text", steps=8)
    (pos, fwd, up), trace, declared, notes = pol.fly(
        _Rend(), ep, pol.goal(ep), clb.geometry_distance)

    assert declared == 2, "should stop on the third reply's Complete"
    assert len(notes) == 3 and notes[-1]["status"] == "Complete"
    # one render for the start pose plus one after each executed command (2 executed, not 3)
    assert [t for t, _ in rendered] == [0, 1, 2]
    assert len(trace) == 3
    # vx=1 for dt=0.5 along forward (-x) moved it 0.5 m toward the subject
    assert rendered[1][1][0] == pytest.approx(5.5, abs=1e-6)
    assert not np.allclose(pos, ep.start.camera_position), "pose never advanced"
    assert all(np.isfinite(trace)), "non-finite distance in the trace"


def test_fly_stops_before_executing_the_completing_command(monkeypatch):
    """`Complete` ends the run; the drone must not also fly that command (V12 shoot semantics)."""
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "clb", "scripts/closed_loop_eval_baselines.py")
    clb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clb)

    class _StubVLM:
        def __init__(self, *a, **k):
            self.calls = self.failures = 0

        def ask_json(self, system, content):
            self.calls += 1
            return {"vx": 5.0, "status": "Complete", "scratchpad": "done"}

    monkeypatch.setattr(clb, "GatewayVLM", _StubVLM)

    class _Rend:
        def render(self, t, pos, fwd, up, want_profile=False):
            return "/tmp/x.jpg"

    gf = SimpleNamespace(raw={"subject_bearing_deg": 0.0}, camera_position=[4, 0, 2],
                         camera_forward=[-1, 0, 0], camera_up=[0, 0, 1],
                         subject_center=[0, 0, 1], subject_height=1.8)
    ep = SimpleNamespace(key="k", name="n", scene="s", goal_frame=gf,
                         start=SimpleNamespace(camera_position=[6, 0, 2],
                                               camera_forward=[-1, 0, 0], camera_up=[0, 0, 1]))
    pol = clb.AstraPolicy("gpt-6-astra", steps=8)
    (pos, _, _), trace, declared, _ = pol.fly(_Rend(), ep, pol.goal(ep), clb.geometry_distance)
    assert declared == 0 and len(trace) == 1
    assert np.allclose(pos, [6, 0, 2]), "the completing command must not be executed"
