from __future__ import annotations

import json
from typing import Mapping, Sequence

from src.scoring.bbox_control import V5_SCORE_KEYS
from src.scoring.evaluator import (
    DEFAULT_TARGET_SCORE_KEYS,
    ALL_SUPPORTED_SCORE_KEYS,
    extract_target_scores,
    normalize_score_value,
)

SCORE_KEYS = list(DEFAULT_TARGET_SCORE_KEYS)
_V5_KEY_SET = set(V5_SCORE_KEYS)


def scores_to_canonical_json(
    scores: Mapping[str, float],
    score_keys: Sequence[str] | None = None,
    decimals: int = 6,
) -> str:
    ordered_keys = list(SCORE_KEYS if score_keys is None else score_keys)
    ordered: dict[str, float | int] = {}
    for key in ordered_keys:
        value = scores[key]
        if key in _V5_KEY_SET:
            ordered[key] = int(round(float(value)))
        else:
            ordered[key] = round(float(value), decimals)
    return json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))


def parse_scores_from_text(
    text: str,
    score_keys: Sequence[str] | None = None,
) -> dict[str, float | int] | None:
    ordered_keys = list(SCORE_KEYS if score_keys is None else score_keys)

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    scores: dict[str, float | int] = {}
    for key in ordered_keys:
        if key not in payload:
            return None
        if key not in ALL_SUPPORTED_SCORE_KEYS:
            return None
        scores[key] = normalize_score_value(key, float(payload[key]))
    return scores


def extract_scores_from_annotation(
    annotation: Mapping[str, object],
    image_width: int,
    image_height: int,
    score_keys: Sequence[str] | None = None,
) -> dict[str, float]:
    ordered_keys = list(SCORE_KEYS if score_keys is None else score_keys)
    return extract_target_scores(
        annotation=dict(annotation),
        image_width=image_width,
        image_height=image_height,
        target_keys=ordered_keys,
    )
