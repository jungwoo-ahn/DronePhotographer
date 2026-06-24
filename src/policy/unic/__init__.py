"""UNIC baseline (issue #22) — aesthetic/composition model used reactively.

Wraps the vendored UNIC composition model (Beyond Image Borders, ICCV 2023): it
recommends a composition box for the current view, which `policy.py` turns into a
camera move. Faithful goal-agnostic baseline. See REFERENCES.md.
"""

from src.policy.unic.model import UNICModel, UNICRecommendation

__all__ = ["UNICModel", "UNICRecommendation"]
