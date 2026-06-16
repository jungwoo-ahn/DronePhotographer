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

    def generate(self, images, system, user):
        self.seen = {"images": images, "system": system, "user": user}
        return self.text


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
