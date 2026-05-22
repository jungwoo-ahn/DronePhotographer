"""Cosmos-backed goal-conditioned policy with joint next-frame + next-action heads.

Both heads are diffusion-based. Conditioning is the shot-profile goal vector.
See CLAUDE.md for the architectural rationale.
"""

from __future__ import annotations


class CosmosWorldActionPolicy:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("src/policy/model.py is a stub")
