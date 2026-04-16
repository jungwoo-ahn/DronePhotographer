"""Compare projected 3D bbox (v3) vs GroundingDINO detection on rendered images.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/compare_bbox_v3_vs_dino.py \
        --run_dir outputs/DogWalk_v3_compare_260318_084442 \
        --caption "a snowman"
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
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="output image path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ann_path = run_dir / "annotations.json"
    annotations = json.loads(ann_path.read_text())

    # Init GroundingDINO
    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for ann in annotations:
        img_path = run_dir / ann["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip: {img_path}")
            continue

        # GroundingDINO detection
        detections = detector.detect(img, args.caption, args.box_threshold, args.text_threshold)

        # Draw projected 3D bbox (green)
        if "bbox_2d" in ann:
            bbox_proj = ann["bbox_2d"]
            elev = ann.get("elevation_deg", 0)
            azi = ann.get("azimuth_deg", 0)
            vis = ann.get("visibility_ratio", 0)
            label_proj = f"v3 proj (e={elev:.0f} a={azi:.0f} v={vis:.2f})"
            draw_bbox(img, bbox_proj, (0, 255, 0), label_proj, thickness=3)

        # Draw GroundingDINO bbox (red)
        for det in detections:
            d = det.as_dict()
            label_dino = f"DINO {d['label']} ({d['score']:.2f})"
            draw_bbox(img, d["bbox_xyxy"], (0, 0, 255), label_dino, thickness=2)

        if not detections:
            cv2.putText(img, "DINO: no detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Add image index
        idx_str = Path(ann["image"]).stem
        cv2.putText(img, idx_str, (img.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        panels.append(img)

    if not panels:
        print("No panels to draw.")
        return

    # Arrange in grid: 2 columns, 5 rows for 10 images
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    h, w = panels[0].shape[:2]
    # Pad to fill grid
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))

    grid_rows = []
    for r in range(rows):
        row_imgs = panels[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    canvas = np.vstack(grid_rows)

    # Add legend at bottom
    legend_h = 40
    legend = np.zeros((legend_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(legend, (20, 10), (50, 30), (0, 255, 0), -1)
    cv2.putText(legend, "Projected 3D bbox (v3)", (60, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(legend, (400, 10), (430, 30), (0, 0, 255), -1)
    cv2.putText(legend, "GroundingDINO detection", (440, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([canvas, legend])

    output_path = args.output or str(run_dir / "compare_v3_vs_dino.png")
    cv2.imwrite(output_path, canvas)
    print(f"Saved comparison to {output_path}")
    print(f"  Canvas size: {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"  Images: {n}")


if __name__ == "__main__":
    main()
