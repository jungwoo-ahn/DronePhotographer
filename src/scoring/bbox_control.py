from __future__ import annotations

from typing import Any, Iterable

BBoxXYXY = tuple[float, float, float, float]


RULE_BASED_SCORE_KEYS = [
    "bbox_occupancy_ratio",
    "bbox_margin_top",
    "bbox_margin_bottom",
    "bbox_margin_left",
    "bbox_margin_right",
    "bbox_aspect_ratio",
    "bbox_centroid_offset",
]


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _box_area(box: BBoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _to_boxes_and_scores(detections: Iterable[Any] | None) -> tuple[list[BBoxXYXY], list[float]]:
    boxes: list[BBoxXYXY] = []
    scores: list[float] = []
    if detections is None:
        return boxes, scores

    for det in detections:
        if isinstance(det, dict):
            box = det.get("bbox_xyxy") or det.get("box_xyxy") or det.get("bbox")
            score = float(det.get("score", 0.0))
        else:
            box = getattr(det, "bbox_xyxy", None) or getattr(det, "box_xyxy", None)
            score = float(getattr(det, "score", 0.0))

        if box is None or len(box) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append((x1, y1, x2, y2))
        scores.append(score)

    return boxes, scores


def _select_primary_box(boxes: list[BBoxXYXY], scores: list[float]) -> BBoxXYXY | None:
    if not boxes:
        return None

    if len(scores) != len(boxes):
        best_idx = max(range(len(boxes)), key=lambda idx: _box_area(boxes[idx]))
        return boxes[best_idx]

    best_idx = max(range(len(boxes)), key=lambda idx: _box_area(boxes[idx]) * scores[idx])
    return boxes[best_idx]


def zero_rule_based_scores() -> dict[str, float]:
    return {key: 0.0 for key in RULE_BASED_SCORE_KEYS}


def compute_rule_based_scores(
    image_width: int,
    image_height: int,
    detections: Iterable[Any] | None,
) -> dict[str, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")

    boxes, scores = _to_boxes_and_scores(detections)
    primary = _select_primary_box(boxes, scores)
    if primary is None:
        return zero_rule_based_scores()

    x1, y1, x2, y2 = primary
    width = float(image_width)
    height = float(image_height)

    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))

    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)

    occupancy = clamp01((bbox_w * bbox_h) / (width * height))

    margin_top = clamp01(y1 / height)
    margin_bottom = clamp01((height - y2) / height)
    margin_left = clamp01(x1 / width)
    margin_right = clamp01((width - x2) / width)

    aspect_ratio = 0.0 if bbox_h == 0.0 else (bbox_w / bbox_h)

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    dx = (cx / width) - 0.5
    dy = (cy / height) - 0.5
    max_center_distance = 0.5 * (2 ** 0.5)
    centroid_offset = clamp01(((dx * dx + dy * dy) ** 0.5) / max_center_distance)

    return {
        "bbox_occupancy_ratio": occupancy,
        "bbox_margin_top": margin_top,
        "bbox_margin_bottom": margin_bottom,
        "bbox_margin_left": margin_left,
        "bbox_margin_right": margin_right,
        "bbox_aspect_ratio": float(aspect_ratio),
        "bbox_centroid_offset": centroid_offset,
    }
