"""Training loop for the Cosmos goal-conditioned policy.

Joint loss over the next-frame diffusion head and the next-action diffusion head,
conditioned on a shot-profile goal. See CLAUDE.md for context.
"""

from __future__ import annotations


class PolicyTrainer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("src/policy/trainer.py is a stub")
