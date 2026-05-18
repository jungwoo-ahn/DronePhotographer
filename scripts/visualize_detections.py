from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--min_score", type=float, default=0.0)
    return parser.parse_args()


def _get_box(det: dict) -> list[float] | None:
    box = det.get("bbox_xyxy")
    if box is None:
        box = det.get("box_xyxy")
    if box is None:
        box = det.get("bbox")
    if box is None:
        return None
    if len(box) != 4:
        return None
    return [float(v) for v in box]


def _draw_detections(image, detections: list[dict], min_score: float) -> int:
    drawn = 0
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < min_score:
            continue

        box = _get_box(det)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label = str(det.get("label", "object"))
        text = f"{label} {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations_path)
    image_root = Path(args.image_root) if args.image_root is not None else annotations_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    start = max(0, int(args.start_idx))
    end = len(annotations) if int(args.limit) <= 0 else min(len(annotations), start + int(args.limit))
    subset = annotations[start:end]

    total_drawn = 0
    for idx, item in enumerate(tqdm(subset, desc="visualize"), start=start):
        image_rel = str(item["image"])
        image_path = image_root / image_rel
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")

        detections = item.get("detections") or []
        drawn = _draw_detections(image=image, detections=detections, min_score=float(args.min_score))
        total_drawn += drawn

        out_name = Path(image_rel).name
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), image)

    print(
        json.dumps(
            {
                "annotations_path": str(annotations_path),
                "image_root": str(image_root),
                "output_dir": str(output_dir),
                "num_images_visualized": len(subset),
                "total_boxes_drawn": int(total_drawn),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""
Example usage:

python scripts/visualize_detections.py \
  --annotations_path outputs/DogWalk_v2_10k_260309_101152/annotations_detected.json \
  --image_root outputs/DogWalk_v2_10k_260309_101152 \
  --output_dir outputs/DogWalk_v2_10k_260309_101152/detection_viz \
  --limit 20
"""
