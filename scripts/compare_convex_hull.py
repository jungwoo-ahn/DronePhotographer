"""Visualize convex hull of projected vertices vs GroundingDINO.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_convex_hull.py \
        --run_dir outputs/DogWalk_v3_compare_260318_084442
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors.detector import GroundingDINODetector


def draw_bbox(img, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_convex_hull(img, points_2d, color, label, alpha=0.15):
    """Draw filled convex hull with transparency."""
    pts = np.array(points_2d, dtype=np.float32)
    # Clip to reasonable range to avoid numerical issues
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], -w, 2 * w)
    pts[:, 1] = np.clip(pts[:, 1], -h, 2 * h)

    try:
        hull = ConvexHull(pts)
    except Exception:
        return

    hull_pts = pts[hull.vertices].astype(np.int32)

    # Semi-transparent fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [hull_pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Outline
    cv2.polylines(img, [hull_pts], isClosed=True, color=color, thickness=2)

    # Label at top-left of hull
    min_x = hull_pts[:, 0].min()
    min_y = hull_pts[:, 1].min()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    ty = max(min_y - 6, th + 4)
    tx = max(min_x, 0)
    cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (tx + 2, ty - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--caption", default="a snowman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    annotations = json.loads((run_dir / "annotations.json").read_text())
    vert_projs = json.loads((run_dir / "vertex_projections.json").read_text())

    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for ann, vp in zip(annotations, vert_projs):
        img_path = run_dir / ann["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1. Convex hull of projected vertices (cyan, semi-transparent)
        pts_2d = vp["projected_2d"]
        if len(pts_2d) >= 3:
            draw_convex_hull(img, pts_2d, (255, 255, 0), "Convex hull", alpha=0.12)

        # 2. GroundingDINO (red)
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
    cv2.rectangle(legend, (20, 10), (50, 30), (255, 255, 0), -1)
    cv2.putText(legend, "Convex hull (all vertices projected)", (60, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(legend, (550, 10), (580, 30), (0, 0, 255), -1)
    cv2.putText(legend, "GroundingDINO", (590, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    canvas = np.vstack([canvas, legend])

    out = str(run_dir / "compare_convex_hull.png")
    cv2.imwrite(out, canvas)
    print(f"Saved: {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
