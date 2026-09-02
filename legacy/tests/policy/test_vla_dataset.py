"""VLADroneDataset shape/keys test on a synthetic v7 placement."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from src.policy.common.action_repr import ACTION_DIM
from src.policy.vla.dataset import VLADroneDataset
from tests.policy.test_v7_integration import _build_dummy_v7_placement


@pytest.fixture
def placement(tmp_path):
    return _build_dummy_v7_placement(tmp_path / "scene_a__obj_x", n_accepted_pairs=2, frames_per_pair=32)


def test_emits_only_vla_keys(placement):
    ds = VLADroneDataset([placement], chunk_size=8, stride=2, target_resolution=(32, 48))
    assert len(ds) > 0
    s = ds[0]
    assert set(s) == {"state_image", "goal_vec", "action_chunk", "meta"}
    # explicitly NOT carrying the world-model-only fields
    assert "next_state_image" not in s
    assert "value_target" not in s


def test_shapes_and_ranges(placement):
    ds = VLADroneDataset([placement], chunk_size=8, stride=2, target_resolution=(32, 48))
    s = ds[0]
    assert s["state_image"].shape == (3, 32, 48)
    assert -1.0 <= float(s["state_image"].min()) and float(s["state_image"].max()) <= 1.0
    assert s["action_chunk"].shape == (8, ACTION_DIM)
    assert bool((s["action_chunk"].abs() <= 1.0).all())   # normalized
    assert s["goal_vec"].ndim == 1


def test_same_window_count_as_base(placement):
    # the VLA dataset must see the identical sample set as the cosmos dataset
    ds = VLADroneDataset([placement], chunk_size=8, stride=1)
    assert len(ds) == len(ds.base)


def test_default_goal_keys_are_all_eight(placement):
    ds = VLADroneDataset([placement], chunk_size=8, stride=4)
    assert ds[0]["goal_vec"].shape == (8,)
