"""Robust parsing + validation of the LLM Policy's action response.

The model is asked for a strict JSON object with dx/dy/dz/dyaw_deg/dpitch_deg.
Real LLMs wrap it in prose or markdown, use trailing commas or all-single-quotes,
or omit fields. `parse_action_response` extracts and lightly repairs the JSON,
then validates the action fields — returning the parsed dict plus diagnostics so
the caller can log response quality and decide whether to retry. This is
llm_policy-local on purpose; the shared `src.vlm.api.parse_vlm_json` is left as-is.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

ACTION_FIELDS = ("dx", "dy", "dz", "dyaw_deg", "dpitch_deg")


@dataclass
class ParseResult:
    parsed: dict | None        # extracted JSON object (None if unrecoverable)
    ok: bool                   # all 5 action fields present and finite
    method: str                # raw | fence | braces (+"+repaired") | none
    missing: list[str]         # action fields absent / non-finite / non-numeric
    reasoning: str | None      # optional model rationale, for logs


def _load(s: str):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def _repair(s: str) -> str:
    """Light, safe repairs for common LLM JSON quirks."""
    s = re.sub(r",\s*([}\]])", r"\1", s)        # trailing commas before } or ]
    if '"' not in s and "'" in s:               # all-single-quoted object
        s = s.replace("'", '"')
    return s


def _candidates(text: str):
    """Yield (label, substring) JSON candidates, most-specific last."""
    yield "raw", text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        yield "fence", m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        yield "braces", text[start : end + 1]


def parse_action_response(text: str) -> ParseResult:
    """Extract + validate the action JSON from raw model text."""
    if not isinstance(text, str):
        text = ""
    parsed, method = None, "none"
    for label, cand in _candidates(text):
        obj = _load(cand)
        if isinstance(obj, dict):
            parsed, method = obj, label
            break
        rep = _repair(cand)
        if rep != cand:
            obj = _load(rep)
            if isinstance(obj, dict):
                parsed, method = obj, f"{label}+repaired"
                break

    if parsed is None:
        return ParseResult(None, False, "none", list(ACTION_FIELDS), None)

    missing: list[str] = []
    for f in ACTION_FIELDS:
        try:
            if not math.isfinite(float(parsed[f])):
                missing.append(f)
        except (KeyError, TypeError, ValueError):
            missing.append(f)
    reasoning = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), str) else None
    return ParseResult(parsed, not missing, method, missing, reasoning)


__all__ = ["ParseResult", "ACTION_FIELDS", "parse_action_response"]
