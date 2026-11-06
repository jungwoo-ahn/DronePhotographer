"""AstraAdapter on the shared contract, against a stubbed gateway (no network, no Blender)."""
import json

import numpy as np
import pytest

from src.baselines.adapters import AstraAdapter, CHUNK_DIM
from src.baselines.astra import GatewayVLM, apply_velocity, clamp_command
from src.common.action_repr import apply_action_9d
from src.common.dataset_base import arrival_step
from src.common.goal_space import DEFAULT_GOAL_KEYS


def _stub_post(replies):
    """A gateway that returns `replies` in order, with usage, as the real one does."""
    it = iter(replies)

    def post(body):
        r = next(it)
        if isinstance(r, int):                      # an HTTP error status
            return r, {}, {}
        n_img = sum(b["type"] == "image_url" for b in body["messages"][1]["content"])
        return 200, {"id": "req", "choices": [{"message": {"content": json.dumps(r)},
                                               "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 500 + 71 * n_img, "completion_tokens": 120,
                               "completion_tokens_details": {"reasoning_tokens": 40}}}, {}
    return post


def _adapter(tmp_path, replies, mode="text"):
    vlm = GatewayVLM("gpt-6-astra", post=_stub_post(replies))
    a = AstraAdapter("gpt-6-astra", mode, tmp_path, vlm=vlm)
    a.begin_episode("Scene__obj:p00:s01:g02")
    return a


POSE = {"position": [1.0, -2.0, 1.6], "forward": [0.0, 1.0, -0.2], "up": [0.0, 0.2, 1.0]}
FRAME = np.full((768, 1024, 3), 128, np.uint8)
GOAL_VEC = np.zeros(len(DEFAULT_GOAL_KEYS), np.float32)
CMD = {"vx": 1.0, "vy": 0.5, "vz": -0.4, "yaw_rate_deg_s": 30.0, "pitch_rate_deg_s": -10.0,
       "status": "Searching", "scratchpad": "subject ahead-left"}


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def test_chunk_decodes_to_the_velocity_pose(tmp_path):
    a = _adapter(tmp_path, [CMD])
    chunk = a.act(FRAME, GOAL_VEC, {}, seed=0, pose=POSE)
    assert chunk.shape == (1, CHUNK_DIM) and chunk[0, 9] == 0.0
    pos, fwd, up = (np.asarray(POSE[k], np.float32) for k in ("position", "forward", "up"))
    fwd, up = _unit(fwd).astype(np.float32), _unit(up - np.dot(up, _unit(fwd)) * _unit(fwd)).astype(np.float32)
    want = apply_velocity(pos, fwd, up, clamp_command(CMD))
    got = apply_action_9d(pos, fwd, up, chunk[0, :9])
    for w, g in zip(want, got):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), atol=2e-4)


def test_complete_is_a_shoot_with_no_motion(tmp_path):
    a = _adapter(tmp_path, [{**CMD, "status": "Complete"}])
    chunk = a.act(FRAME, GOAL_VEC, {}, seed=0, pose=POSE)
    assert arrival_step(chunk[:, 9]) == 0          # the loop executes chunk[:0] and halts
    assert not chunk[0, :9].any()


def test_frames_sent_at_296x222_and_context_accumulates(tmp_path):
    from PIL import Image
    a = _adapter(tmp_path, [CMD] * 14)
    for t in range(14):
        a.act(FRAME, GOAL_VEC, {}, seed=t, pose=POSE)
    rec = a.episode_record()
    assert rec["n_calls"] == 14
    # current frame + min(t, 12) history frames
    assert [c["n_images"] for c in rec["calls"]] == [1 + min(t, 12) for t in range(14)]
    assert Image.open(a._ep["dir"] / "t00.jpg").size == (296, 222)
    assert rec["prompt_tokens"] == sum(500 + 71 * (1 + min(t, 12)) for t in range(14))
    assert rec["cost_usd"] == pytest.approx(rec["prompt_tokens"] * 10e-6 + 14 * 120 * 50e-6)


def test_image_goal_goes_first_and_is_resized(tmp_path):
    from PIL import Image
    g = tmp_path / "goal_src.jpg"
    Image.fromarray(FRAME).save(g)
    a = _adapter(tmp_path, [CMD], mode="image")
    a.act(FRAME, GOAL_VEC, {}, seed=0, pose=POSE, goal_image=str(g))
    assert a.episode_record()["calls"][0]["n_images"] == 2
    assert Image.open(a._ep["dir"] / "goal.jpg").size == (296, 222)


def test_transport_errors_retry_then_count(tmp_path, monkeypatch):
    monkeypatch.setattr("src.baselines.astra.time.sleep", lambda s: None)
    a = _adapter(tmp_path, [429, 503, CMD])
    a.act(FRAME, GOAL_VEC, {}, seed=0, pose=POSE)
    c = a.episode_record()["calls"][0]
    assert [x["status"] for x in c["attempts"]] == [429, 503, 200] and c["parsed"]


def test_unparseable_reply_hovers(tmp_path):
    a = _adapter(tmp_path, ["not json at all"])
    chunk = a.act(FRAME, GOAL_VEC, {}, seed=0, pose=POSE)
    assert a.episode_record()["n_parse_failed"] == 1
    np.testing.assert_allclose(chunk[0, :3], 0, atol=1e-6)       # no translation
    assert chunk[0, 9] == 0.0                                       # and keeps flying
