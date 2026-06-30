"""Reverse-trajectory augmentation: balances the dataset's far->near (dolly-in) bias.

A forward window's camera path is replayed backward, yielding the dolly-OUT motion the
data lacks (~81% of trajectories dolly in) with no re-render. These tests confirm the
reversed windows are the inverse direction and that the dataset flag actually adds them.
"""
import os

import numpy as np
import pytest

from src.policy.common.annotations import iter_windows
from src.policy.common.dataset_base import BasePolicyDataset, _compute_action_chunk

DATA = "data/trajectories"


def _a_placement() -> str:
    if not os.path.isdir(DATA):
        pytest.skip("no data/trajectories")
    for n in sorted(os.listdir(DATA)):
        if os.path.exists(f"{DATA}/{n}/data.json"):
            return f"{DATA}/{n}/data.json"
    pytest.skip("no data.json under data/trajectories")


def test_reverse_same_count_and_reversed_order():
    dj = _a_placement()
    fwd = list(iter_windows(dj, chunk_size=8, stride=4))
    rev = list(iter_windows(dj, chunk_size=8, stride=4, reverse=True))
    assert len(fwd) > 0 and len(rev) == len(fwd)        # one reverse per forward window
    # a reversed window starts at a LATER trajectory frame and ends at an EARLIER one
    assert all(w.start_frame_idx > w.end_frame_idx for w in rev)
    assert all(w.start_frame_idx < w.end_frame_idx for w in fwd)


def test_reverse_flips_dolly_direction():
    dj = _a_placement()
    fwd = list(iter_windows(dj, chunk_size=8, stride=4))
    rev = list(iter_windows(dj, chunk_size=8, stride=4, reverse=True))
    # action[:,2] is Δforward (the dolly axis); its net sign flips between the two passes
    fdir = np.mean([_compute_action_chunk(w)[:, 2].sum() for w in fwd])
    rdir = np.mean([_compute_action_chunk(w)[:, 2].sum() for w in rev])
    assert fdir * rdir < 0, f"expected opposite dolly directions, got fwd={fdir:.3f} rev={rdir:.3f}"


def test_augment_reverse_adds_windows():
    dj = _a_placement()
    base = BasePolicyDataset([dj], chunk_size=8, stride=4, augment_reverse=False)
    aug = BasePolicyDataset([dj], chunk_size=8, stride=4, augment_reverse=True)
    # reversed windows roughly double the dataset (some reverse goals near the
    # trajectory's far end are clamped/off-frame and get filtered, so not exactly 2x)
    assert len(aug) > len(base) * 1.3
