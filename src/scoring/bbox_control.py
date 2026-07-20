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

    # Near-plane blow-up handling. When the dolly closes in, a grazing mesh
    # vertex at depth ~0 projects to +/-thousands of px and inflates the
    # mesh-tight AABB (often in a single axis). This is a legitimate extreme
    # close-up (the subject fills or overflows the view), NOT an off-subject
    # frame. The old VLM-era policy zeroed the whole profile here to avoid
    # "5-digit pixel coords the VLM can't predict"; for the geometric goal
    # space we instead read bounded geometry off the FRAME-CLIPPED bbox so the
    # close-up becomes a valid, distinct goal (occupancy stays high, center /
    # offset stay inside the frame). `normalize_goal` still lets genuinely
    # off-frame (but finite) subjects read |n|>1 via the normal branch below.
    img_w, img_h = float(image_width), float(image_height)
    img_area = img_w * img_h
    extreme = (
        bbox_full_w > 4.0 * img_w
        or bbox_full_h > 4.0 * img_h
        or full_area > 16.0 * img_area
    )

    cx1 = max(0.0, min(x1f, img_w))
    cx2 = max(0.0, min(x2f, img_w))
    cy1 = max(0.0, min(y1f, img_h))
    cy2 = max(0.0, min(y2f, img_h))
    clipped_w = max(0.0, cx2 - cx1)
    clipped_h = max(0.0, cy2 - cy1)
    clipped_area = clipped_w * clipped_h

    if extreme:
        # Geometry from the clipped bbox (bounded to the frame); body_in_frame
        # from the full area capped at the blow-up threshold so a single grazing
        # vertex cannot drive the ratio to ~0.
        cx = (cx1 + cx2) * 0.5
        cy = (cy1 + cy2) * 0.5
        x_offset = int(round(clipped_w * 0.5))
        y_offset = int(round(clipped_h * 0.5))
        denom = min(bbox_full_w, 4.0 * img_w) * min(bbox_full_h, 4.0 * img_h)
    else:
        cx = (x1f + x2f) * 0.5
        cy = (y1f + y2f) * 0.5
        x_offset = int(round(bbox_full_w * 0.5))
        y_offset = int(round(bbox_full_h * 0.5))
        denom = full_area

    occupancy_pct = max(0, min(100, int(round(100.0 * clipped_area / img_area))))
    if denom > 0.0:
        body_in_frame_pct = max(0, min(100, int(round(100.0 * clipped_area / denom))))
    else:
        body_in_frame_pct = 0

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
