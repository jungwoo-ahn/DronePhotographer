from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.scoring.evaluator import ALL_SUPPORTED_SCORE_KEYS

DEFAULT_SCORE_WEIGHTS = {
    "bbox_occupancy_ratio": 2.0,
    "bbox_margin_top": 1.0,
    "bbox_margin_bottom": 1.0,
    "bbox_margin_left": 1.0,
    "bbox_margin_right": 1.0,
    "bbox_aspect_ratio": 1.0,
    "bbox_centroid_offset": 2.0,
}

TARGET_PRESETS: dict[str, dict[str, float]] = {
    "centered_50": {
        "bbox_centroid_offset": 0.0,
        "bbox_occupancy_ratio": 0.5,
    },
    "centered_square_medium": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.3,
        "aspect_ratio": 1.0,
    },
    "centered_square_close": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.55,
        "aspect_ratio": 1.0,
    },
    "top_right_thirds_medium": {
        "center_x": 2.0 / 3.0,
        "center_y": 1.0 / 3.0,
        "occupancy": 0.25,
        "aspect_ratio": 1.0,
    },
    "centered_wide_18": {
        "bbox_centroid_offset": 0.0,
        "bbox_aspect_ratio": 1.8,
    },
    "cinematic_center_wide": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.35,
        "aspect_ratio": 1.8,
    },
    "centered_wide_close_18": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.48,
        "aspect_ratio": 1.8,
    },
    "centered_portrait_medium": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.3,
        "aspect_ratio": 0.67,
    },
    "centered_portrait_close": {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.4,
        "aspect_ratio": 0.67,
    },
    "top_right_thirds_portrait_medium": {
        "center_x": 2.0 / 3.0,
        "center_y": 1.0 / 3.0,
        "occupancy": 0.2,
        "aspect_ratio": 0.67,
    },
}


@dataclass(frozen=True)
class TargetObjective:
    name: str
    raw_spec: dict[str, float]
    score_targets: dict[str, float]
    score_weights: dict[str, float]


def available_target_presets() -> list[str]:
    return sorted(TARGET_PRESETS)


def preset_specs() -> dict[str, dict[str, float]]:
    return {name: dict(spec) for name, spec in TARGET_PRESETS.items()}


def _load_json_dict(text_or_path: str) -> dict[str, float]:
    text_or_path = str(text_or_path).strip()
    candidate_path = Path(text_or_path)
    if candidate_path.exists():
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text_or_path)
    if not isinstance(payload, dict):
        raise ValueError("target/weight JSON must decode to an object")
    return {str(key): float(value) for key, value in payload.items()}


def composition_to_margin_targets(
    center_x: float,
    center_y: float,
    occupancy: float,
    aspect_ratio: float,
    frame_aspect_ratio: float = 4.0 / 3.0,
) -> dict[str, float]:
    center_x = float(center_x)
    center_y = float(center_y)
    occupancy = float(occupancy)
    aspect_ratio = float(aspect_ratio)

    if occupancy < 0.0 or occupancy > 1.0:
        raise ValueError("occupancy must be in [0, 1]")
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be > 0")
    if frame_aspect_ratio <= 0.0:
        raise ValueError("frame_aspect_ratio must be > 0")
    if center_x < 0.0 or center_x > 1.0 or center_y < 0.0 or center_y > 1.0:
        raise ValueError("center_x and center_y must be in [0, 1]")

    norm_ar = aspect_ratio / frame_aspect_ratio
    width = math.sqrt(occupancy * norm_ar)
    height = math.sqrt(occupancy / norm_ar)

    left = center_x - 0.5 * width
    right = 1.0 - center_x - 0.5 * width
    top = center_y - 0.5 * height
    bottom = 1.0 - center_y - 0.5 * height

    margins = {
        "bbox_margin_top": top,
        "bbox_margin_bottom": bottom,
        "bbox_margin_left": left,
        "bbox_margin_right": right,
    }
    min_margin = min(margins.values())
    if min_margin < -1e-6:
        raise ValueError(
            "composition target does not fit inside the frame; reduce occupancy or move the target center"
        )
    return {key: max(0.0, float(value)) for key, value in margins.items()}


def compile_score_targets(
    spec: Mapping[str, float],
    frame_aspect_ratio: float = 4.0 / 3.0,
) -> dict[str, float]:
    raw_spec = {str(key): float(value) for key, value in spec.items()}
    targets: dict[str, float] = {}

    for key in ALL_SUPPORTED_SCORE_KEYS:
        if key in raw_spec:
            targets[key] = float(raw_spec[key])

    if "occupancy" in raw_spec:
        targets["bbox_occupancy_ratio"] = float(raw_spec["occupancy"])
    if "aspect_ratio" in raw_spec:
        targets["bbox_aspect_ratio"] = float(raw_spec["aspect_ratio"])

    has_center = "center_x" in raw_spec or "center_y" in raw_spec
    if has_center:
        if "center_x" not in raw_spec or "center_y" not in raw_spec:
            raise ValueError("center_x and center_y must be provided together")
        center_x = float(raw_spec["center_x"])
        center_y = float(raw_spec["center_y"])
        centered = abs(center_x - 0.5) < 1e-6 and abs(center_y - 0.5) < 1e-6
        if centered and "bbox_centroid_offset" not in targets:
            targets["bbox_centroid_offset"] = 0.0
        if "occupancy" in raw_spec and "aspect_ratio" in raw_spec:
            targets.update(
                composition_to_margin_targets(
                    center_x=center_x,
                    center_y=center_y,
                    occupancy=float(raw_spec["occupancy"]),
                    aspect_ratio=float(raw_spec["aspect_ratio"]),
                    frame_aspect_ratio=frame_aspect_ratio,
                )
            )
        elif not centered:
            raise ValueError(
                "off-center composition targets require occupancy and aspect_ratio "
                "so bbox margins can be derived"
            )

    if not targets:
        raise ValueError(
            "target spec did not produce any score targets; provide bbox score keys directly "
            "or use center_x/center_y/occupancy/aspect_ratio"
        )
    return targets


def compile_score_weights(
    score_targets: Mapping[str, float],
    weights_spec: Mapping[str, float] | None = None,
) -> dict[str, float]:
    weights_spec = {} if weights_spec is None else {str(key): float(value) for key, value in weights_spec.items()}
    score_weights: dict[str, float] = {}
    for key in score_targets:
        weight = float(weights_spec.get(key, DEFAULT_SCORE_WEIGHTS.get(key, 1.0)))
        if weight < 0.0:
            raise ValueError(f"score weight must be non-negative for key '{key}'")
        score_weights[key] = weight
    return score_weights


def weighted_target_error(
    predicted_scores: Mapping[str, float],
    score_targets: Mapping[str, float],
    score_weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    total_weight = 0.0
    total_error = 0.0
    per_key_errors: dict[str, float] = {}
    for key, target_value in score_targets.items():
        weight = float(score_weights.get(key, 1.0))
        if weight <= 0.0:
            continue
        predicted_value = float(predicted_scores[key])
        error_value = abs(predicted_value - float(target_value))
        per_key_errors[key] = error_value
        total_error += weight * error_value
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0, per_key_errors
    return total_error / total_weight, per_key_errors


def build_target_objective(
    *,
    preset_name: str | None = None,
    target_json_text: str | None = None,
    score_weights_json_text: str | None = None,
    frame_aspect_ratio: float = 4.0 / 3.0,
) -> TargetObjective:
    raw_spec: dict[str, float] = {}
    if preset_name:
        if preset_name not in TARGET_PRESETS:
            raise ValueError(
                f"unknown target preset '{preset_name}'. Available presets: {', '.join(available_target_presets())}"
            )
        raw_spec.update(TARGET_PRESETS[preset_name])
    if target_json_text:
        raw_spec.update(_load_json_dict(target_json_text))

    score_targets = compile_score_targets(raw_spec, frame_aspect_ratio=frame_aspect_ratio)
    weight_spec = _load_json_dict(score_weights_json_text) if score_weights_json_text else None
    score_weights = compile_score_weights(score_targets, weights_spec=weight_spec)
    objective_name = preset_name if preset_name else "custom_target"
    return TargetObjective(
        name=objective_name,
        raw_spec=raw_spec,
        score_targets=score_targets,
        score_weights=score_weights,
    )
