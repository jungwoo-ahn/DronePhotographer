"""LLMPhotoPolicy tests with a mock backend (no model/API loaded)."""

import numpy as np
import pytest

from src.policy.common.action_repr import ACTION_DIM
from src.policy.llm_policy.backends import OpenAIBackend, VLMBackend, build_backend
from src.policy.llm_policy.policy import LLMPhotoPolicy
from src.policy.llm_policy.prompt import build_user_prompt, describe_goal


class _MockBackend(VLMBackend):
    def __init__(self, text):
        self.text = text
        self.seen = None
        self.calls = 0

    def generate(self, images, system, user):
        self.seen = {"images": images, "system": system, "user": user}
        self.calls += 1
        return self.text


class _SeqBackend(VLMBackend):
    """Returns a different response per call (to exercise retries)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def generate(self, images, system, user):
        self.calls += 1
        return self.texts[min(self.calls - 1, len(self.texts) - 1)]


_TARGET = {"occupancy": 35, "cam_to_obj_azimuth_deg": 180, "cam_to_obj_elevation_deg": 0,
           "object_center_x": 512, "object_center_y": 384}


def test_act_parses_json_to_5d_action_with_deg_to_rad():
    txt = '{"reasoning":"pan","dx":0.5,"dy":-0.2,"dz":1.0,"dyaw_deg":30,"dpitch_deg":-10}'
    pol = LLMPhotoPolicy(_MockBackend(txt))
    action, info = pol.act(object(), _TARGET)
    assert action.shape == (ACTION_DIM,)
    np.testing.assert_allclose(
        action, [0.5, -0.2, 1.0, np.deg2rad(30), np.deg2rad(-10)], atol=1e-5)
    assert info["parsed"]["reasoning"] == "pan"


def test_act_handles_markdown_fenced_json():
    txt = "Sure!\n```json\n{\"dx\":0,\"dy\":0,\"dz\":2.0,\"dyaw_deg\":0,\"dpitch_deg\":0}\n```"
    pol = LLMPhotoPolicy(_MockBackend(txt))
    action, _ = pol.act(object(), _TARGET)
    assert action[2] == pytest.approx(2.0)


def test_act_clamps_excessive_moves():
    txt = '{"dx":99,"dy":-99,"dz":0,"dyaw_deg":999,"dpitch_deg":-999}'
    pol = LLMPhotoPolicy(_MockBackend(txt), max_translation_m=3.0, max_angle_deg=60.0)
    action, _ = pol.act(object(), _TARGET)
    assert action[0] == pytest.approx(3.0) and action[1] == pytest.approx(-3.0)
    assert action[3] == pytest.approx(np.deg2rad(60.0)) and action[4] == pytest.approx(-np.deg2rad(60.0))


def test_act_unparseable_response_is_zero_action():
    pol = LLMPhotoPolicy(_MockBackend("I cannot help with that."))
    action, info = pol.act(object(), _TARGET)
    assert np.count_nonzero(action) == 0
    assert info["parsed"] is None


def test_act_repairs_trailing_comma_and_prose():
    txt = 'Here is the move:\n{"dx": 0.1, "dy": 0.0, "dz": 0.5, "dyaw_deg": 5, "dpitch_deg": 0,}'
    pol = LLMPhotoPolicy(_MockBackend(txt))
    action, info = pol.act(object(), _TARGET)
    assert info["ok"] and "repaired" in info["method"]
    assert action[2] == pytest.approx(0.5)


def test_act_repairs_single_quoted_object():
    txt = "{'dx': 0.0, 'dy': 0.0, 'dz': 1.5, 'dyaw_deg': 0, 'dpitch_deg': 0}"
    pol = LLMPhotoPolicy(_MockBackend(txt))
    action, info = pol.act(object(), _TARGET)
    assert info["ok"] and action[2] == pytest.approx(1.5)


def test_act_retries_on_missing_fields_then_succeeds():
    bad = '{"reasoning": "thinking", "dx": 0.2}'           # missing dy/dz/dyaw/dpitch
    good = '{"dx": 0.2, "dy": 0.0, "dz": 0.3, "dyaw_deg": 0, "dpitch_deg": 0}'
    be = _SeqBackend([bad, good])
    pol = LLMPhotoPolicy(be, max_retries=1)
    action, info = pol.act(object(), _TARGET)
    assert be.calls == 2 and info["retries"] == 1 and info["ok"]
    assert action[2] == pytest.approx(0.3)


def test_act_reports_missing_fields_when_retries_exhausted():
    bad = '{"dx": 0.2}'
    pol = LLMPhotoPolicy(_MockBackend(bad), max_retries=1)
    action, info = pol.act(object(), _TARGET)
    assert not info["ok"]
    assert set(info["missing"]) == {"dy", "dz", "dyaw_deg", "dpitch_deg"}
    assert action[0] == pytest.approx(0.2)   # present field still used; rest zero
    assert np.count_nonzero(action[1:]) == 0


def test_describe_goal_is_human_readable():
    desc = describe_goal(_TARGET)
    assert "35%" in desc
    assert "centered" in desc           # object_center_x/y at 512/384 -> centered
    assert "eye level" in desc          # elevation 0
    user = build_user_prompt(_TARGET)
    assert "JSON" in user or "json" in user
    assert "Desired framing" in user


def test_build_backend_selects_and_validates():
    assert isinstance(build_backend({"backend": "api", "api": {"api_base": "x", "model": "m"}}), OpenAIBackend)
    with pytest.raises(ValueError):
        build_backend({"backend": "nonsense"})
