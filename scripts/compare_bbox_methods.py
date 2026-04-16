"""Visualize 3 bbox methods: AABB projection vs vertex projection vs GroundingDINO.

Runs inside Blender for vertex access, then calls GroundingDINO separately.
This script runs outside Blender - reads pre-rendered images + projection_test.json,
runs GroundingDINO, and draws all 3 bboxes.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_bbox_methods.py \
        --run_dir outputs/DogWalk_v3_compare_260318_084442
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors.detector import GroundingDINODetector


def draw_bbox(img, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--caption", default="a snowman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    annotations = json.loads((run_dir / "annotations.json").read_text())
    proj_test = json.loads((run_dir / "projection_test.json").read_text())

    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for ann, proj in zip(annotations, proj_test):
        img_path = run_dir / ann["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1. AABB projection (green) - from annotations
        if ann.get("bbox_2d"):
            draw_bbox(img, ann["bbox_2d"], (0, 255, 0), "AABB 8-corner", thickness=2)

        # 2. Vertex projection (yellow) - from projection_test
        if proj.get("bbox_verts"):
            draw_bbox(img, proj["bbox_verts"], (0, 255, 255), "All vertices", thickness=3)

        # 3. GroundingDINO (red)
        detections = detector.detect(img, args.caption, 0.25, 0.20)
        for det in detections:
            d = det.as_dict()
            draw_bbox(img, d["bbox_xyxy"], (0, 0, 255), f"DINO ({d['score']:.2f})", thickness=2)
        if not detections:
            cv2.putText(img, "DINO: none", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        idx_str = Path(ann["image"]).stem
        cv2.putText(img, idx_str, (img.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)

    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))

    grid_rows = [np.hstack(panels[r * cols:(r + 1) * cols]) for r in range(rows)]
    canvas = np.vstack(grid_rows)

    legend_h = 40
    legend = np.zeros((legend_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(legend, (20, 10), (50, 30), (0, 255, 0), -1)
    cv2.putText(legend, "AABB 8-corner", (60, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(legend, (300, 10), (330, 30), (0, 255, 255), -1)
    cv2.putText(legend, "All vertices (tight)", (340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(legend, (620, 10), (650, 30), (0, 0, 255), -1)
    cv2.putText(legend, "GroundingDINO", (660, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    canvas = np.vstack([canvas, legend])

    out = str(run_dir / "compare_3methods.png")
    cv2.imwrite(out, canvas)
    print(f"Saved: {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
