from __future__ import annotations

from typing import Any, Iterable, Sequence

from .bbox_control import RULE_BASED_SCORE_KEYS, compute_rule_based_scores
from .subject_aware import SUBJECT_AWARE_SCORE_KEYS

CAMERA_3D_SCORE_KEYS = [
    "camera_to_object_fx", "camera_to_object_fy", "camera_to_object_fz",
    "camera_to_object_ux", "camera_to_object_uy", "camera_to_object_uz",
]

TOP_LEVEL_SCORE_KEYS = ["body_in_frame_ratio"]

_CAMERA_3D_KEY_TO_IDX = {k: i for i, k in enumerate(CAMERA_3D_SCORE_KEYS)}

DEFAULT_TARGET_SCORE_KEYS = list(RULE_BASED_SCORE_KEYS)
ALL_SUPPORTED_SCORE_KEYS = (
    list(RULE_BASED_SCORE_KEYS)
    + list(SUBJECT_AWARE_SCORE_KEYS)
    + list(CAMERA_3D_SCORE_KEYS)
    + list(TOP_LEVEL_SCORE_KEYS)
)


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
    if score_key in TOP_LEVEL_SCORE_KEYS:
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

    camera_3d_keys = [key for key in target_keys if key in CAMERA_3D_SCORE_KEYS]
    top_level_keys = [key for key in target_keys if key in TOP_LEVEL_SCORE_KEYS]
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

    if camera_3d_keys:
        c2o = annotation.get("camera_to_object_6d")
        if c2o is None:
            raise KeyError(
                "missing 'camera_to_object_6d' in annotation. "
                "Re-render with object orientation or remove camera_to_object_* from target_score_keys."
            )
        for key in camera_3d_keys:
            out[key] = float(c2o[_CAMERA_3D_KEY_TO_IDX[key]])

    for key in top_level_keys:
        value = annotation.get(key)
        if value is None:
            out[key] = 0.0  # missing = not visible
        else:
            out[key] = normalize_score_value(key, float(value))

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
