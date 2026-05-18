"""Visualize mask-based bbox vs GroundingDINO vs vertex convex hull.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_mask_bbox.py \
        --run_dir outputs/DogWalk_v3_mask_test \
        --hull_dir outputs/DogWalk_v3_compare_260318_084442
"""
import argparse, json, sys
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
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_contour(img, mask, color, label, alpha=0.15):
    """Draw mask contour and semi-transparent fill."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.drawContours(img, contours, -1, color, 2)
    # Label
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        tx, ty = max(int(xs.min()), 0), max(int(ys.min()) - 6, 14)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
        cv2.putText(img, label, (tx + 2, ty - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--caption", default="a snowman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    mask_dir = run_dir / "masks"

    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for i in range(10):
        img_path = run_dir / f"images/img_{i:04d}.png"
        mask_path = mask_dir / f"mask_{i:04d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1. Mask-based silhouette (green contour + fill)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            draw_contour(img, mask_bin, (0, 255, 0), "ID mask silhouette", alpha=0.12)
            # Mask-based bbox
            ys, xs = np.where(mask_bin > 0)
            if len(xs) > 0:
                bbox_mask = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                draw_bbox(img, bbox_mask, (0, 255, 0), "mask bbox", thickness=2)

        # 2. GroundingDINO (red)
        dets = detector.detect(img, args.caption, 0.25, 0.20)
        for d in dets:
            dd = d.as_dict()
            draw_bbox(img, dd["bbox_xyxy"], (0, 0, 255), f"DINO ({dd['score']:.2f})", thickness=2)
        if not dets:
            cv2.putText(img, "DINO: none", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"img_{i:04d}", (img.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        panels.append(img)

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))
    grid = np.vstack([np.hstack(panels[r*cols:(r+1)*cols]) for r in range(rows)])

    leg = np.zeros((45, grid.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(leg, (20, 12), (48, 30), (0, 255, 0), -1)
    cv2.putText(leg, "ID mask silhouette + bbox (pixel-perfect)", (56, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(leg, (530, 12), (558, 30), (0, 0, 255), -1)
    cv2.putText(leg, "GroundingDINO", (566, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    canvas = np.vstack([grid, leg])

    out = str(run_dir / "compare_mask_vs_dino.png")
    cv2.imwrite(out, canvas)
    print(f"Saved: {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
