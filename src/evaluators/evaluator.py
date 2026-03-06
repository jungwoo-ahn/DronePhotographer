from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import inspect
import math
import threading
import logging

BBoxXYXY = Tuple[float, float, float, float]



# EVALUATION RESULT DATACLASS

@dataclass(frozen=True)
class EvaluationResult:
    name: str
    score: float
    details: Dict[str, Any]



# HELPERS

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d == 0:
        return default
    return n / d


def _box_area(box: BBoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box: BBoxXYXY) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pick_primary_box(
    boxes: Sequence[BBoxXYXY],
    scores: Optional[Sequence[float]] = None,
) -> Optional[Tuple[BBoxXYXY, Optional[float]]]:
    if not boxes:
        return None
    if scores is None or len(scores) != len(boxes):
        best_idx = max(range(len(boxes)), key=lambda i: _box_area(boxes[i]))
        return boxes[best_idx], None
    best_idx = max(range(len(boxes)), key=lambda i: float(scores[i]) * _box_area(boxes[i]))
    return boxes[best_idx], float(scores[best_idx])


def _as_boxes_and_scores(detections: Optional[Iterable[Any]]) -> Tuple[List[BBoxXYXY], List[float]]:
    """Accepts either:
    - list of dicts like {bbox_xyxy: [x1,y1,x2,y2], score: 0.9}
    - list of objects like detector.Detection with .box_xyxy and .score
    """
    if detections is None:
        return [], []
    boxes: List[BBoxXYXY] = []
    scores: List[float] = []
    for d in detections:
        if isinstance(d, dict):
            box = d.get("bbox_xyxy") or d.get("box_xyxy") or d.get("bbox")
            sc = d.get("score", 0.0)
        else:
            box = getattr(d, "bbox_xyxy", None) or getattr(d, "box_xyxy", None)
            sc = getattr(d, "score", 0.0)
        if box is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append((x1, y1, x2, y2))
        scores.append(float(sc))
    return boxes, scores


def _is_pathlike(value: Any) -> bool:
    import os

    return isinstance(value, (str, bytes, os.PathLike))


def _to_gray(image_bgr):
    import cv2

    if image_bgr is None:
        return None
    if _is_pathlike(image_bgr):
        return cv2.imread(str(image_bgr), cv2.IMREAD_GRAYSCALE)
    if not hasattr(image_bgr, "shape"):
        return None
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_pil_rgb(image_bgr):
    from PIL import Image

    if image_bgr is None:
        return None
    if _is_pathlike(image_bgr):
        try:
            return Image.open(image_bgr).convert("RGB")
        except Exception:
            return None
    if not hasattr(image_bgr, "shape"):
        return None
    if len(image_bgr.shape) == 2:
        try:
            return Image.fromarray(image_bgr).convert("RGB")
        except Exception:
            return None
    try:
        import cv2
    except Exception:
        return None
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None



# BASE EVALUATOR CLASS

class Evaluator:
    name: str = "evaluator"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        raise NotImplementedError



# SUBJECT COMPOSITION EVALUATORS

class ColorContrastEvaluator(Evaluator):
    """Scores local contrast inside subject vs background.

    Heuristic:
    - Convert to grayscale
    - Compare stddev inside subject bbox vs outside
    - Higher difference => better separation
    """

    name = "color_contrast"

    def __init__(self, min_subject_area_frac: float = 0.01) -> None:
        self.min_subject_area_frac = float(min_subject_area_frac)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np
        import cv2

        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)
        if area_frac < self.min_subject_area_frac:
            return EvaluationResult(self.name, 0.0, {"reason": "subject_too_small", "area_frac": area_frac})

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        mask[y1:y2, x1:x2] = 255

        subj = gray[mask == 255]
        bg = gray[mask == 0]
        if subj.size < 50 or bg.size < 50:
            return EvaluationResult(self.name, 0.0, {"reason": "insufficient_pixels"})

        subj_std = float(subj.std())
        bg_std = float(bg.std())
        # normalized separation: |subj-bg| / (subj+bg+eps)
        sep = abs(subj_std - bg_std) / (subj_std + bg_std + 1e-6)
        score = _clamp01(sep)
        return EvaluationResult(
            self.name,
            score,
            {
                "subject_std": subj_std,
                "background_std": bg_std,
                "separation": sep,
                "det_score": det_score,
                "area_frac": area_frac,
            },
        )


class SubjectSizeEvaluator(Evaluator):

    name = "subject_size"

    def __init__(self, target_area_frac: float = 0.18, tolerance: float = 0.18) -> None:
        self.target_area_frac = float(target_area_frac)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)

        # Triangular score peaked at target_area_frac
        dist = abs(area_frac - self.target_area_frac)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"area_frac": area_frac, "target": self.target_area_frac, "tolerance": self.tolerance, "det_score": det_score},
        )

class SubjectSizeEvaluator_80(Evaluator):
    
    name = "subject_size_80"

    def __init__(self, target_area_frac: float = 0.8, tolerance: float = 0.8) -> None:
        self.target_area_frac = float(target_area_frac)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)

        # Triangular score peaked at target_area_frac
        dist = abs(area_frac - self.target_area_frac)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"area_frac": area_frac, "target": self.target_area_frac, "tolerance": self.tolerance, "det_score": det_score},
        )
        
class SubjectSizeEvaluator_50(Evaluator):
    
    name = "subject_size_50"

    def __init__(self, target_area_frac: float = 0.5, tolerance: float = 0.5) -> None:
        self.target_area_frac = float(target_area_frac)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)

        # Triangular score peaked at target_area_frac
        dist = abs(area_frac - self.target_area_frac)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"area_frac": area_frac, "target": self.target_area_frac, "tolerance": self.tolerance, "det_score": det_score},
        )
        
class SubjectSizeEvaluator_20(Evaluator):
    
    name = "subject_size_20"

    def __init__(self, target_area_frac: float = 0.2, tolerance: float = 0.8) -> None:
        self.target_area_frac = float(target_area_frac)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)

        # Triangular score peaked at target_area_frac
        dist = abs(area_frac - self.target_area_frac)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"area_frac": area_frac, "target": self.target_area_frac, "tolerance": self.tolerance, "det_score": det_score},
        )

class RuleOfThirdsEvaluator(Evaluator):
    """Scores how close the subject center is to rule-of-thirds intersections."""
    # or ... consider only the horizontal distance from vertical thirds lines? #TODO

    name = "rule_of_thirds"

    def __init__(self, softness: float = 0.35) -> None:
        # softness controls how quickly score decays with distance
        self.softness = float(softness)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        cx, cy = _box_center(box)

        # 4 intersection points
        pts = [
            (w / 3.0, h / 3.0),
            (2.0 * w / 3.0, h / 3.0),
            (w / 3.0, 2.0 * h / 3.0),
            (2.0 * w / 3.0, 2.0 * h / 3.0),
        ]
        dists = [math.hypot((cx - px) / w, (cy - py) / h) for px, py in pts]
        dmin = min(dists)
        # Map distance in normalized units to [0,1] with exponential falloff
        score = math.exp(-_safe_div(dmin, self.softness, 1.0))
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"center": [cx, cy], "dmin_norm": dmin, "softness": self.softness, "det_score": det_score},
        )


class CenterednessEvaluator(Evaluator):
    """Scores how close the subject center is to the image center."""

    name = "centeredness"

    def __init__(self, softness: float = 0.35) -> None:
        # softness controls how quickly score decays with distance
        self.softness = float(softness)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        cx, cy = _box_center(box)

        dx = (cx - (w / 2.0)) / w
        dy = (cy - (h / 2.0)) / h
        d = math.hypot(dx, dy)

        score = math.exp(-_safe_div(d, self.softness, 1.0))
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"center": [cx, cy], "d_norm": d, "softness": self.softness, "det_score": det_score},
        )


class RuleOfThirdsEvaluator_Line(Evaluator):
    """Scores how close the subject center is to rule-of-thirds lines based on box orientation."""

    name = "rule_of_thirds_line"

    def __init__(self, softness: float = 0.35) -> None:
        self.softness = float(softness)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1
        cx, cy = _box_center(box)

        if h_box > w_box:
            # vertical lines
            lines = [w / 3.0, 2.0 * w / 3.0]
            dists = [abs(cx - lx) / w for lx in lines]
            dmin = min(dists)
            orientation = "vertical"
        else:
            # horizontal lines
            lines = [h / 3.0, 2.0 * h / 3.0]
            dists = [abs(cy - ly) / h for ly in lines]
            dmin = min(dists)
            orientation = "horizontal"

        score = math.exp(-_safe_div(dmin, self.softness, 1.0))
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"center": [cx, cy], "dmin_norm": dmin, "softness": self.softness, "det_score": det_score, "orientation": orientation},
        )


class BreathingSpaceEvaluator(Evaluator):
    """Scores breathing space by the largest edge margin.
    Larger max margin => higher score.
    """

    name = "breathing_space"

    def __init__(self, min_margin_frac: float = 0.05, target_margin_frac: float = 0.12) -> None:
        self.min_margin_frac = float(min_margin_frac)
        self.target_margin_frac = float(target_margin_frac)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        x1, y1, x2, y2 = box

        left = _safe_div(x1, w, 0.0)
        right = _safe_div(w - x2, w, 0.0)
        top = _safe_div(y1, h, 0.0)
        bottom = _safe_div(h - y2, h, 0.0)
        min_margin = min(left, right, top, bottom)
        max_margin = max(left, right, top, bottom)
        score = _clamp01(max_margin)
        return EvaluationResult(
            self.name,
            score,
            {
                "margins": {"left": left, "right": right, "top": top, "bottom": bottom},
                "min_margin": min_margin,
                "max_margin": max_margin,
                "min_margin_frac": self.min_margin_frac,
                "target_margin_frac": self.target_margin_frac,
                "det_score": det_score,
            },
        )



class BBoxMarginTopEvaluator(Evaluator):
    """Scores top margin linearly: margin=0 -> score=0, margin=1 -> score=1."""

    name = "bbox_margin_top"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        margin_frac = _clamp01(_safe_div(box[1], h, 0.0))
        return EvaluationResult(self.name, margin_frac, {"margin_frac": margin_frac, "det_score": det_score})


class BBoxMarginBottomEvaluator(Evaluator):
    """Scores bottom margin linearly: margin=0 -> score=0, margin=1 -> score=1."""

    name = "bbox_margin_bottom"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        margin_frac = _clamp01(_safe_div(h - box[3], h, 0.0))
        return EvaluationResult(self.name, margin_frac, {"margin_frac": margin_frac, "det_score": det_score})


class BBoxMarginLeftEvaluator(Evaluator):
    """Scores left margin linearly: margin=0 -> score=0, margin=1 -> score=1."""

    name = "bbox_margin_left"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        margin_frac = _clamp01(_safe_div(box[0], w, 0.0))
        return EvaluationResult(self.name, margin_frac, {"margin_frac": margin_frac, "det_score": det_score})


class BBoxMarginRightEvaluator(Evaluator):
    """Scores right margin linearly: margin=0 -> score=0, margin=1 -> score=1."""

    name = "bbox_margin_right"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        margin_frac = _clamp01(_safe_div(w - box[2], w, 0.0))
        return EvaluationResult(self.name, margin_frac, {"margin_frac": margin_frac, "det_score": det_score})


class BBoxAspectRatioEvaluator(Evaluator):
    """Scores how close the bbox aspect ratio (w/h) is to a target ratio."""

    name = "bbox_aspect_ratio"

    def __init__(self, target_ratio: float = 1.0, tolerance: float = 0.5) -> None:
        self.target_ratio = float(target_ratio)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        ratio = _safe_div(bw, bh, 0.0)
        dist = abs(ratio - self.target_ratio)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name, score,
            {"ratio": ratio, "target": self.target_ratio, "tolerance": self.tolerance, "bbox_w": bw, "bbox_h": bh, "det_score": det_score},
        )


class BBoxAspectRatio4x3Evaluator(BBoxAspectRatioEvaluator):
    name = "bbox_aspect_ratio_4_3"

    def __init__(self, tolerance: float = 0.35) -> None:
        super().__init__(target_ratio=4.0 / 3.0, tolerance=tolerance)


class BBoxAspectRatio16x9Evaluator(BBoxAspectRatioEvaluator):
    name = "bbox_aspect_ratio_16_9"

    def __init__(self, tolerance: float = 0.4) -> None:
        super().__init__(target_ratio=16.0 / 9.0, tolerance=tolerance)


class BBoxAspectRatio3x2Evaluator(BBoxAspectRatioEvaluator):
    name = "bbox_aspect_ratio_3_2"

    def __init__(self, tolerance: float = 0.35) -> None:
        super().__init__(target_ratio=3.0 / 2.0, tolerance=tolerance)


class BBoxAspectRatio1x1Evaluator(BBoxAspectRatioEvaluator):
    name = "bbox_aspect_ratio_1_1"

    def __init__(self, tolerance: float = 0.3) -> None:
        super().__init__(target_ratio=1.0, tolerance=tolerance)


class BBoxAspectRatio9x16Evaluator(BBoxAspectRatioEvaluator):
    name = "bbox_aspect_ratio_9_16"

    def __init__(self, tolerance: float = 0.25) -> None:
        super().__init__(target_ratio=9.0 / 16.0, tolerance=tolerance)


class BBoxAspectRatioCommonEvaluator(Evaluator):
    """Scores bbox ratio by nearest common aspect ratio preset."""

    name = "bbox_aspect_ratio_common"

    def __init__(self, tolerance: float = 0.35) -> None:
        self.tolerance = float(tolerance)
        self.targets: Dict[str, float] = {
            "1:1": 1.0,
            "4:3": 4.0 / 3.0,
            "3:2": 3.0 / 2.0,
            "16:9": 16.0 / 9.0,
            "9:16": 9.0 / 16.0,
        }

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})

        box, det_score = picked
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        ratio = _safe_div(bw, bh, 0.0)

        nearest_label = ""
        nearest_target = 0.0
        nearest_dist = float("inf")
        for label, target in self.targets.items():
            dist = abs(ratio - target)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = label
                nearest_target = target

        score = 1.0 - _safe_div(nearest_dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {
                "ratio": ratio,
                "nearest_ratio": nearest_label,
                "nearest_target": nearest_target,
                "distance": nearest_dist,
                "tolerance": self.tolerance,
                "bbox_w": bw,
                "bbox_h": bh,
                "det_score": det_score,
            },
        )


"""LOW-LEVEL EVALUATORS

class BrightnessEvaluator(Evaluator):
    # Calculates the mean brightness of the image (0-255).

    name = "brightness"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(np.mean(gray))
        return EvaluationResult(self.name, score, {"mean": score})


class LaplacianEvaluator(Evaluator):
    # Calculates the variance of the Laplacian (edge detection).

    name = "laplacian"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import cv2

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return EvaluationResult(self.name, score, {"variance": score})


class StddevEvaluator(Evaluator):
    # Calculates the standard deviation of pixel intensities.

    name = "stddev"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(np.std(gray))
        return EvaluationResult(self.name, score, {"stddev": score})
"""



# AI-BASED EVALUATORS

_QALIGN_MODEL = None
_QALIGN_LOAD_ERROR: Optional[str] = None
_QALIGN_LOCK = threading.Lock()


def _patch_qalign_runtime_compat(model: Any) -> None:
    # Some remote QAlign model revisions expect internal Llama flags that are
    # not present depending on transformers version.
    try:
        for module in model.modules():
            cls_name = module.__class__.__name__.lower()
            if "llama" not in cls_name:
                continue
            if not hasattr(module, "_use_flash_attention_2"):
                setattr(module, "_use_flash_attention_2", False)
            if not hasattr(module, "_use_sdpa"):
                setattr(module, "_use_sdpa", False)
    except Exception:
        pass


def _patch_transformers_rotary_compat() -> None:
    # Compatibility patch for remote QAlign code that calls
    # LlamaRotaryEmbedding(..., seq_len=...) without position_ids.
    try:
        import torch
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        orig_forward = LlamaRotaryEmbedding.forward
        sig = inspect.signature(orig_forward)
        position_param = sig.parameters.get("position_ids")
        if position_param is None or position_param.default is not inspect._empty:
            return
        if getattr(LlamaRotaryEmbedding.forward, "_qalign_compat_patched", False):
            return

        def _forward_with_optional_position_ids(self, x, position_ids=None, seq_len=None):
            if position_ids is None:
                bsz, q_len = x.shape[0], x.shape[-2]
                position_ids = torch.arange(q_len, device=x.device).unsqueeze(0).expand(bsz, -1)
            return orig_forward(self, x, position_ids=position_ids, seq_len=seq_len)

        _forward_with_optional_position_ids._qalign_compat_patched = True
        LlamaRotaryEmbedding.forward = _forward_with_optional_position_ids
    except Exception:
        pass


def _load_qalign_model():
    global _QALIGN_MODEL, _QALIGN_LOAD_ERROR
    if _QALIGN_MODEL is not None:
        return _QALIGN_MODEL
    with _QALIGN_LOCK:
        if _QALIGN_MODEL is not None:
            return _QALIGN_MODEL
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            _QALIGN_LOAD_ERROR = f"import_failed: {exc}"
            logging.warning("QAlign import failed: %s", exc)
            return None

        _patch_transformers_rotary_compat()

        # Some QAlign remote model revisions reference typing symbols without
        # importing them, which raises NameError at module import time.
        try:
            import builtins
            from transformers.cache_utils import Cache as _HFCache
            from transformers.modeling_outputs import (
                BaseModelOutputWithPast as _HFBaseModelOutputWithPast,
                CausalLMOutputWithPast as _HFCausalLMOutputWithPast,
            )
            from transformers.utils import logging as _hf_logging

            if not hasattr(builtins, "Cache"):
                builtins.Cache = _HFCache
            if not hasattr(builtins, "BaseModelOutputWithPast"):
                builtins.BaseModelOutputWithPast = _HFBaseModelOutputWithPast
            if not hasattr(builtins, "CausalLMOutputWithPast"):
                builtins.CausalLMOutputWithPast = _HFCausalLMOutputWithPast
            if not hasattr(builtins, "logger"):
                builtins.logger = _hf_logging.get_logger("qalign_remote")
        except Exception:
            pass

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            def _from_pretrained_with_dtype_kwargs(use_device_map: bool):
                common = dict(
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                if use_device_map:
                    common["device_map"] = "auto"

                try:
                    return AutoModelForCausalLM.from_pretrained(
                        "q-future/one-align",
                        dtype=dtype,
                        **common,
                    )
                except TypeError:
                    return AutoModelForCausalLM.from_pretrained(
                        "q-future/one-align",
                        torch_dtype=dtype,
                        **common,
                    )

            try:
                _QALIGN_MODEL = _from_pretrained_with_dtype_kwargs(use_device_map=True)
            except Exception as first_exc:
                # Fallback path when accelerate/device_map auto is unavailable.
                logging.warning("QAlign auto device_map failed, retrying without device_map: %s", first_exc)
                _QALIGN_MODEL = _from_pretrained_with_dtype_kwargs(use_device_map=False)
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                _QALIGN_MODEL = _QALIGN_MODEL.to(target_device)

            _patch_qalign_runtime_compat(_QALIGN_MODEL)
            _QALIGN_MODEL.eval()
        except Exception as exc:
            _QALIGN_LOAD_ERROR = f"load_failed: {exc}"
            logging.warning("QAlign load failed: %s", exc)
            _QALIGN_MODEL = None
        return _QALIGN_MODEL


class _QAlignBaseEvaluator(Evaluator):
    def __init__(self, task: str, normalize: bool = True) -> None:
        self.task = str(task)
        self.normalize = bool(normalize)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        from PIL import Image

        pil = _to_pil_rgb(image_bgr)
        if pil is None and hasattr(image_bgr, "shape"):
            try:
                if len(image_bgr.shape) == 2:
                    pil = Image.fromarray(image_bgr).convert("RGB")
                else:
                    pil = Image.fromarray(image_bgr[:, :, ::-1]).convert("RGB")
            except Exception:
                pil = None

        if pil is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})

        model = _load_qalign_model()
        if model is None:
            details = {"reason": "model_load_failed"}
            if _QALIGN_LOAD_ERROR:
                details["error"] = _QALIGN_LOAD_ERROR
            return EvaluationResult(self.name, 0.0, details)

        try:
            import torch

            with torch.no_grad():
                score_tensor = model.score([pil], task_=self.task, input_="image")
            raw_score = float(score_tensor.item()) if hasattr(score_tensor, "item") else float(score_tensor[0])
        except Exception as exc:
            logging.warning("QAlign scoring failed: %s", exc)
            return EvaluationResult(self.name, 0.0, {"reason": "score_failed", "error": str(exc)})

        score = raw_score
        if self.normalize:
            score = _clamp01((raw_score - 1.0) / 4.0)

        return EvaluationResult(
            self.name,
            score,
            {
                "raw_score": raw_score,
                "task": self.task,
                "normalized": self.normalize,
            },
        )


class QAlignEvaluator(_QAlignBaseEvaluator):
    name = "qalign"

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(task="quality", normalize=normalize)


class QAlignQualityEvaluator(_QAlignBaseEvaluator):
    name = "qalign_quality"

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(task="quality", normalize=normalize)


class QAlignAestheticEvaluator(_QAlignBaseEvaluator):
    name = "qalign_aesthetic"

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(task="aesthetics", normalize=normalize)



# COMPOSITE EVALUATOR

class CompositeEvaluator(Evaluator):
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        weights: Optional[Sequence[float]] = None,
        name: str = "composite",
    ) -> None:
        self.name = name
        self.evaluators = list(evaluators)
        if weights is None:
            self.weights = [1.0] * len(self.evaluators)
        else:
            if len(weights) != len(self.evaluators):
                raise ValueError("weights must match evaluators length")
            self.weights = [float(w) for w in weights]

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        results = [e.evaluate(image_bgr, detections) for e in self.evaluators]
        wsum = sum(self.weights)
        score = 0.0
        if wsum > 0:
            score = sum(w * r.score for w, r in zip(self.weights, results)) / wsum
        details: Dict[str, Any] = {
            "components": [{"name": r.name, "score": r.score, "details": r.details} for r in results],
            "weights": self.weights,
        }
        return EvaluationResult(name=self.name, score=_clamp01(score), details=details)

def default_composition_evaluator() -> CompositeEvaluator:
    return CompositeEvaluator(
        evaluators=[
            ColorContrastEvaluator(),
            SubjectSizeEvaluator(),
            RuleOfThirdsEvaluator(),
            BreathingSpaceEvaluator(),            
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
        name="composition",
    )


# example 1. aesthetic image, medium shot (1/4 of the whole image), with good breathing space
def test_evaluator_example_1():
    return CompositeEvaluator(
        evaluators=[
            QAlignAestheticEvaluator(normalize=True),
            SubjectSizeEvaluator(target_area_frac=0.25, tolerance=0.5),
            BreathingSpaceEvaluator(min_margin_frac=0.08, target_margin_frac=0.15),
        ],
        weights=[0.6, 0.3, 0.1],
        name="aesthetic_ms_breathing",
    )

# example 2. aesthetic image, following rule of thirds.
def test_evaluator_example_2():
    return CompositeEvaluator(
        evaluators=[
            QAlignAestheticEvaluator(normalize=True),
            RuleOfThirdsEvaluator(softness=0.3),
        ],
        weights=[0.7, 0.3],
        name="aesthetic_rot",
    )
    
# example 3. aesthetic image, extreme close-up shot (2/3 of the whole image), with minimal breathing space
def test_evaluator_example_3():
    return CompositeEvaluator(
        evaluators=[
            QAlignAestheticEvaluator(normalize=True),
            SubjectSizeEvaluator(target_area_frac=0.66, tolerance=0.66),
            BreathingSpaceEvaluator(min_margin_frac=0.01, target_margin_frac=0.05),
        ],
        # weights=[0.6, 0.3, 0.1],
        weights=[45, 45, 10],
        name="aesthetic_ecu_minimal",
    )

if __name__ == "__main__":
    import cv2
    from src.detectors.detector import GroundingDINODetector

    IMAGE_PATH = "outputs/Koky_LuxuryHouse_0_deterministic_radius_4_pos_fixed_251231_102750/images/img_0003.png"
    CONFIG_PATH = "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    CAPTION = "a person"
    image = cv2.imread(IMAGE_PATH)
    model = GroundingDINODetector(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
    detections = model.detect_file(
        image_path=IMAGE_PATH,
        caption=CAPTION,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )       
    
    
    evaluator = default_composition_evaluator()
    result = evaluator.evaluate(image_bgr=image, detections=detections)
    print(f"Composition score: {result.score:.4f}")
    print("Details:", result.details)
    
    """
    Composition score: 0.6325
    Details: {'components': [
    {'name': 'color_contrast', 'score': 0.029035447147387018, 'details': {'subject_std': 68.63666150001279, 'background_std': 72.74164384158124, 'separation': 0.029035447147387018, 'det_score': 0.8605086207389832, 'area_frac': 0.19032145346260432}}, 
    {'name': 'subject_size', 'score': 0.9426585918744205, 'details': {'area_frac': 0.19032145346260432, 'target': 0.18, 'tolerance': 0.18, 'det_score': 0.8605086207389832}}, 
    {'name': 'rule_of_thirds', 'score': 0.5582473096267943, 'details': {'center': [248.23076629638672, 240.64374542236328], 'dmin_norm': 0.20403362265007988, 'softness': 0.35, 'det_score': 0.8605086207389832}}, 
    {'name': 'breathing_space', 'score': 1.0, 'details': {'margins': {'left': 0.36357465386390686, 'right': 0.39392322301864624, 'top': 0.07759538292884827, 'bottom': 0.1375807523727417}, 'min_margin': 0.07759538292884827, 'min_margin_frac': 0.05, 'target_margin_frac': 0.12, 'det_score': 0.8605086207389832}}
    ], 
    'weights': [1.0, 1.0, 1.0, 1.0]}
    """
    

    
# https://www.studiobinder.com/blog/ultimate-guide-to-camera-shots/#ELS

# Shot Types (by Size):
# Extreme Wide Shot (EWS)
# Long Shot (LS)
# Full Shot (FS)
# Medium Wide Shot (MWS)
# Cowboy Shot (CS)
# Medium Shot (MS)
# Medium Close-Up (MCU)
# Close-Up (CU)
# Extreme Close-Up (ECU)

# Shot Types (by Framing):
