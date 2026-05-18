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


V5_SCORE_KEYS = [
    "occupancy",
    "body_in_frame_ratio",
    "cam_to_obj_azimuth_deg",
    "cam_to_obj_elevation_deg",
    "object_center_x",
    "object_center_y",
    "bbox_x_offset",
    "bbox_y_offset",
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


def zero_v5_scores() -> dict[str, int]:
    return {key: 0 for key in V5_SCORE_KEYS}


def compute_v5_scores(
    image_width: int,
    image_height: int,
    bbox_full: BBoxXYXY | None,
    azimuth_deg: float,
    elevation_deg: float,
) -> dict[str, int]:
    """v5 integer score schema.

    bbox_full: full projected bbox in pixel coords; may extend beyond image.
    Returns 8 ints: occupancy, body_in_frame_ratio (0-100 percent),
    cam_to_obj_azimuth_deg/elevation_deg, object_center_x/y (unbounded),
    bbox_x_offset/y_offset (>=0).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")

    az = int(round(float(azimuth_deg))) % 360
    if az < 0:
        az += 360
    el = max(-90, min(90, int(round(float(elevation_deg)))))

    if bbox_full is None:
        out = zero_v5_scores()
        out["cam_to_obj_azimuth_deg"] = az
        out["cam_to_obj_elevation_deg"] = el
        return out

    x1f, y1f, x2f, y2f = (float(v) for v in bbox_full)
    bbox_full_w = max(0.0, x2f - x1f)
    bbox_full_h = max(0.0, y2f - y1f)
    full_area = bbox_full_w * bbox_full_h

    # Clamp pathological projections (camera near-plane blow-ups when a 3D
    # bbox corner is essentially at camera depth = 0). Anything beyond ~4x
    # the image dimension is "fully off-screen"; report it as such instead of
    # leaking 5-digit pixel coordinates that the VLM can't predict.
    extreme_w = bbox_full_w > 4.0 * image_width
    extreme_h = bbox_full_h > 4.0 * image_height
    if extreme_w or extreme_h or full_area > 16.0 * (float(image_width) * float(image_height)):
        out = zero_v5_scores()
        out["cam_to_obj_azimuth_deg"] = az
        out["cam_to_obj_elevation_deg"] = el
        return out

    cx = (x1f + x2f) * 0.5
    cy = (y1f + y2f) * 0.5
    x_offset = int(round(bbox_full_w * 0.5))
    y_offset = int(round(bbox_full_h * 0.5))

    cx1 = max(0.0, min(x1f, float(image_width)))
    cx2 = max(0.0, min(x2f, float(image_width)))
    cy1 = max(0.0, min(y1f, float(image_height)))
    cy2 = max(0.0, min(y2f, float(image_height)))
    clipped_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    img_area = float(image_width) * float(image_height)

    occupancy_pct = int(round(100.0 * clipped_area / img_area))
    if full_area > 0.0:
        body_in_frame_pct = int(round(100.0 * clipped_area / full_area))
    else:
        body_in_frame_pct = 0
    occupancy_pct = max(0, min(100, occupancy_pct))
    body_in_frame_pct = max(0, min(100, body_in_frame_pct))

    return {
        "occupancy": occupancy_pct,
        "body_in_frame_ratio": body_in_frame_pct,
        "cam_to_obj_azimuth_deg": az,
        "cam_to_obj_elevation_deg": el,
        "object_center_x": int(round(cx)),
        "object_center_y": int(round(cy)),
        "bbox_x_offset": max(0, x_offset),
        "bbox_y_offset": max(0, y_offset),
    }
