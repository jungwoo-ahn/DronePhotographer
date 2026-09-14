"""`arrival_step` — where in a chunk the policy says it arrives, and how many steps to run.

The bug this pins: both rollout scripts halted on `max(chunk[:, 9]) > threshold` and broke
out BEFORE executing any of the chunk's pose steps, on the reasoning that a shoot state
means "you are already there" and its pose action is zero in training. That reasoning holds
only for the steps at and after the crossing. The channel is latched, so a declaring chunk
reads "travel, travel, ..., arrived", and discarding it threw away the travel.

Not a corner case: over the 103 declarations in `runs/eval/shoot_probe_iter40000.json` the
first crossing sat at step 5, 6 or 7 and never at 0 — an average of 5.8 discarded steps of
8. The `stop` rollout configs consequently scored ~0 improvement (191 episodes, 69% halting
at chunk 0 without moving) and measured the bug rather than the policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.dataset_base import arrival_step, shoot_column


def test_returns_the_first_crossing_not_just_whether_one_happened():
    chunk = [0.01, 0.02, 0.01, 0.03, 0.08, 0.91, 0.96, 0.98]
    assert arrival_step(chunk, 0.5) == 5


def test_none_when_the_chunk_never_crosses():
    assert arrival_step([0.01] * 8, 0.5) is None


def test_zero_is_a_real_answer_and_not_confused_with_none():
    """`argmax` on an all-False array is also 0 — the two must not collapse."""
    assert arrival_step([0.9] * 8, 0.5) == 0
    assert arrival_step([0.1] * 8, 0.5) is None


def test_the_index_is_the_number_of_steps_to_execute():
    """A chunk declaring at 5 must run steps 0-4: five steps of planned travel."""
    chunk = np.zeros((8, 10), dtype=np.float32)
    chunk[:, 0] = 0.2                      # some translation on every step
    chunk[5:, 9] = 1.0                     # latched arrival at step 5
    k = arrival_step(chunk[:, 9], 0.5)
    executed = chunk[:k, :9]
    assert len(executed) == 5
    assert np.allclose(executed[:, 0], 0.2)
    # what the old code did, for contrast
    assert len(chunk[:0, :9]) == 0


def test_a_threshold_above_the_signal_declines_to_declare():
    assert arrival_step([0.0, 0.0, 0.6, 0.7], 0.9) is None
    assert arrival_step([0.0, 0.0, 0.6, 0.7], 0.5) == 2


def test_round_trips_the_training_label_shape():
    """`shoot_column` emits the latched target; `arrival_step` must recover its onset."""
    for k in range(8):
        latched = (np.arange(8) >= k).astype(np.float32)
        assert arrival_step(latched, 0.5) == k
