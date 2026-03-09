from __future__ import annotations

from typing import Any, Iterable, Sequence

from .bbox_control import RULE_BASED_SCORE_KEYS, compute_rule_based_scores
from .subject_aware import SUBJECT_AWARE_SCORE_KEYS


DEFAULT_TARGET_SCORE_KEYS = list(RULE_BASED_SCORE_KEYS)
ALL_SUPPORTED_SCORE_KEYS = list(RULE_BASED_SCORE_KEYS) + list(SUBJECT_AWARE_SCORE_KEYS)


def _annotation_field_name(score_key: str) -> str:
    return f"score_{score_key}"


def _clamp_subject_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def normalize_score_value(score_key: str, value: float) -> float:
    if score_key in SUBJECT_AWARE_SCORE_KEYS:
        return _clamp_subject_score(value)
    if score_key in RULE_BASED_SCORE_KEYS and score_key != "bbox_aspect_ratio":
        return _clamp_subject_score(value)
    return float(value)


def extract_target_scores(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
    target_keys: Sequence[str],
) -> dict[str, float]:
    target_keys = list(target_keys)
    rule_keys = [key for key in target_keys if key in RULE_BASED_SCORE_KEYS]
    subject_keys = [key for key in target_keys if key in SUBJECT_AWARE_SCORE_KEYS]

    unsupported = [key for key in target_keys if key not in ALL_SUPPORTED_SCORE_KEYS]
    if unsupported:
        raise ValueError(f"unsupported score keys: {unsupported}")

    out: dict[str, float] = {}

    if rule_keys:
        rule_scores = compute_rule_based_scores(
            image_width=image_width,
            image_height=image_height,
            detections=annotation.get("detections"),
        )
        for key in rule_keys:
            out[key] = normalize_score_value(key, float(rule_scores[key]))

    for key in subject_keys:
        field = _annotation_field_name(key)
        if field not in annotation:
            raise KeyError(
                f"missing subject-aware score field '{field}'. "
                f"Add VLM scores first or remove '{key}' from target_score_keys."
            )
        out[key] = normalize_score_value(key, float(annotation[field]))

    return out


def flatten_scores_for_annotation(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    scores = compute_rule_based_scores(
        image_width=image_width,
        image_height=image_height,
        detections=annotation.get("detections"),
    )
    return {f"score_{key}": float(value) for key, value in scores.items()}
