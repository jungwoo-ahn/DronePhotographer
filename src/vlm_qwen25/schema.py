from __future__ import annotations

import json
from typing import Mapping

SCORE_KEYS = [
    "rule_of_thirds_line",
    "breathing_space",
    "centeredness",
    "subject_size_20",
    "subject_size_80",
]

ANNOTATION_SCORE_FIELDS = {
    "rule_of_thirds_line": "score_rule_of_thirds_line",
    "breathing_space": "score_breathing_space",
    "centeredness": "score_centeredness",
    "subject_size_20": "score_subject_size_20",
    "subject_size_80": "score_subject_size_80",
}


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def extract_scores(annotation: Mapping[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in SCORE_KEYS:
        field = ANNOTATION_SCORE_FIELDS[key]
        value = annotation[field]
        scores[key] = clamp01(float(value))
    return scores


def scores_to_canonical_json(scores: Mapping[str, float], decimals: int = 6) -> str:
    ordered = {key: round(float(scores[key]), decimals) for key in SCORE_KEYS}
    return json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))


def parse_scores_from_text(text: str) -> dict[str, float] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    scores: dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in payload:
            return None
        scores[key] = clamp01(float(payload[key]))
    return scores
