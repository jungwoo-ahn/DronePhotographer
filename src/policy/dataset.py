"""Hindsight-relabeled (state, goal-profile, action) dataset.

Loads Blender renders + annotations, pairs consecutive frames into
(state, random action, next-state) transitions, and uses the achieved shot
profile of the next-state as the goal. See CLAUDE.md for the design rationale.
"""

from __future__ import annotations


class HindsightProfileDataset:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("src/policy/dataset.py is a stub")
