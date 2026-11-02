"""Is the subject actually in the picture? Detector checks for rendered frames.

Every stage-3 score is computed from the subject's projected mesh, which cannot see occlusion:
a wall between the camera and the subject still reads as a well-framed shot. Measured on the
held-out data, ~30% of goal frames and 15-18% of start frames show no person at all. These
helpers ask a person detector instead, and compare its box with where the mesh projects, so
"visible" means "the subject is where the geometry says it is", not "some person somewhere".
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_YOLO = str(Path(__file__).resolve().parents[2] / "yolo11l-pose.pt")


def clip_box(box: Sequence[float], width: float, height: float) -> list[float] | None:
    """Clip an UNCLIPPED projection (`bbox_xyxy_full`) to the frame; None if it is all outside.

    The stored box routinely runs off the frame (a y0 of -436 when the legs leave the bottom),
    while a detector can only report what is inside the picture, so compare clipped boxes.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    out = [max(0.0, x0), max(0.0, y0), min(width, x1), min(height, y1)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def subject_luminance(image: str, box: Sequence[float]) -> float:
    """Mean luminance (0-255) inside a clipped subject box."""
    from PIL import Image, ImageStat

    im = Image.open(image).convert("L")
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return float(ImageStat.Stat(im.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))).mean[0])


def detect_people(frames: Iterable[str], model: str = DEFAULT_YOLO, batch: int = 32,
                  conf: float = 0.25, log_every: int = 2000) -> dict[str, list[list[float]]]:
    """Person boxes `[x0, y0, x1, y1, conf]` for every frame, one detector pass each."""
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLO(model)
    frames = list(dict.fromkeys(frames))
    out: dict[str, list[list[float]]] = {}
    for i in range(0, len(frames), batch):
        part = frames[i:i + batch]
        for f, r in zip(part, yolo.predict(part, imgsz=640, conf=conf, classes=[0],
                                           device=device, verbose=False)):
            out[f] = [[float(v) for v in b] + [float(c)]
                      for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        if log_every and (i // batch) % max(1, log_every // batch) == 0:
            print(f"[detect] {min(i + batch, len(frames))}/{len(frames)} on {device}", flush=True)
    return out


def best_iou(boxes: list[list[float]], target: Sequence[float]) -> float:
    return max((iou(b[:4], target) for b in boxes), default=0.0)


__all__ = ["DEFAULT_YOLO", "clip_box", "iou", "subject_luminance", "detect_people", "best_iou"]
