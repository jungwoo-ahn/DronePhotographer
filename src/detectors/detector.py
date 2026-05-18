from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_IMPORT_ERROR: Exception | None = None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GROUNDINGDINO_REPO = _REPO_ROOT / "repos" / "GroundingDINO"

try:
    from groundingdino.util.inference import Model
except Exception as exc:
    _IMPORT_ERROR = exc
    if _GROUNDINGDINO_REPO.exists():
        sys.path.insert(0, str(_GROUNDINGDINO_REPO))
        try:
            from groundingdino.util.inference import Model

            _IMPORT_ERROR = None
        except Exception as inner_exc:
            _IMPORT_ERROR = inner_exc

BBoxXYXY = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: BBoxXYXY

    def as_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.box_xyxy
        return {
            "label": self.label,
            "score": float(self.score),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        }


def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _clip_box_xyxy(box: Iterable[float], width: int, height: int) -> BBoxXYXY:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, float(width - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


class GroundingDINODetector:
    def __init__(
        self,
        model_config_path: str | Path,
        model_checkpoint_path: str | Path,
        device: Optional[str] = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import GroundingDINO. Install with Blender Python: "
                "`SETUPTOOLS_USE_DISTUTILS=stdlib blender/4.5/python/bin/python3.11 "
                "-m pip install --no-build-isolation --no-deps -e repos/GroundingDINO`"
            ) from _IMPORT_ERROR

        self.model_config_path = str(_to_path(model_config_path))
        self.model_checkpoint_path = str(_to_path(model_checkpoint_path))
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self._model = Model(
            model_config_path=self.model_config_path,
            model_checkpoint_path=self.model_checkpoint_path,
            device=device,
        )

    def detect(
        self,
        image_bgr,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        """Run detection on an OpenCV BGR -> return `Detection` """
        if image_bgr is None:
            return []
        h, w = image_bgr.shape[:2]

        detections, phrases = self._model.predict_with_caption(
            image=image_bgr,
            caption=str(caption),
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )

        # GroundingDINO returns a supervision.Detections-like object.
        # We defensively extract fields without hard-coding a single version.
        boxes = getattr(detections, "xyxy", None)
        confidences = getattr(detections, "confidence", None)
        if boxes is None or confidences is None:
            return []

        if phrases is None:
            phrases = ["object"] * len(confidences)

        out: List[Detection] = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            label = phrases[i] if i < len(phrases) else "object"
            out.append(
                Detection(
                    label=str(label),
                    score=float(conf),
                    box_xyxy=_clip_box_xyxy(box, width=w, height=h),
                )
            )
        return out

    def detect_file(
        self,
        image_path: str | Path,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        import cv2

        image_path = _to_path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.detect(
            image_bgr=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )


def annotate_image(image_bgr, detections: Sequence[Detection]):
    # Draw xyxy boxes and labels on a BGR image
    try:
        import supervision as sv
        import numpy as np

        xyxy = np.array([d.box_xyxy for d in detections], dtype=float)
        conf = np.array([d.score for d in detections], dtype=float)
        # supervision expects class_id; we set dummy 0s and encode label ourselves.
        class_id = np.zeros((len(detections),), dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        labels = [f"{d.label} {d.score:.2f}" for d in detections]
        annotator = sv.BoxAnnotator()
        return annotator.annotate(scene=image_bgr.copy(), detections=det, labels=labels)
    except Exception:
        import cv2

        img = image_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d.box_xyxy]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{d.label} {d.score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return img


def main() -> None:
    import argparse
    import json
    import cv2

    parser = argparse.ArgumentParser(description="Run GroundingDINO detection on one image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, required=True, help="GroundingDINO config file")
    parser.add_argument("--weights", type=str, required=True, help="GroundingDINO checkpoint")
    parser.add_argument("--caption", type=str, required=True, help="Text prompt/caption")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu")
    parser.add_argument("--viz_out", type=str, default=None, help="Optional output image path")
    parser.add_argument("--json_out", type=str, default=None, help="Optional output json path")
    args = parser.parse_args()

    if not args.caption.strip():
        raise SystemExit("Provide a non-empty --caption")

    det = GroundingDINODetector(args.config, args.weights, device=args.device)
    detections = det.detect_file(
        args.image,
        caption=args.caption,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.json_out:
        payload = {
            "image": str(args.image),
            "caption": str(args.caption),
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "detections": [d.as_dict() for d in detections],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))

    if args.viz_out:
        image = cv2.imread(str(args.image))
        annotated = annotate_image(image, detections)
        cv2.imwrite(str(args.viz_out), annotated)

    print(json.dumps([d.as_dict() for d in detections], indent=2))


if __name__ == "__main__":
    main()
    
    
"""

CUDA_VISIBLE_DEVICES=6 python detector.py \
  --image outputs/Koky_LuxuryHouse_0_deterministic_radius_8_pos_fixed_251231_102813/images/img_0181.png \
  --config repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights repos/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --caption "a man with black clothes" \
  --viz_out outputs/dino_viz.png \
  --json_out outputs/dino.json
    
"""
