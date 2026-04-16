"""Compare mask bbox tightening methods vs GroundingDINO.

Tests morphological opening with different kernel sizes to find the
tightest bbox that still covers the main body.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_mask_tight.py \
        --run_dir outputs/DogWalk_v3_mask_test
"""
import argparse, sys
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
    cv2.rectangle(img, (tx:=max(x1,0), ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (tx + 2, ty - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_contour(img, mask, color, alpha=0.12):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.drawContours(img, contours, -1, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--caption", default="a snowman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for i in range(10):
        img = cv2.imread(str(run_dir / f"images/img_{i:04d}.png"))
        mask = cv2.imread(str(run_dir / f"masks/mask_{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 1. Raw mask silhouette (thin green outline, no fill)
        raw_bbox = mask_to_bbox(mask_bin)

        # 2. Morphological opening: erode then dilate to remove thin parts
        # Use elliptical kernel, size relative to image
        # Try kernel proportional to image size (~3% of shorter dim)
        short_dim = min(img.shape[:2])
        k_size = max(5, int(short_dim * 0.03)) | 1  # ensure odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask_opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

        # Keep only the largest connected component
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened)
        if n_labels > 1:
            # Skip label 0 (background)
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_opened = ((labels == largest) * 255).astype(np.uint8)

        opened_bbox = mask_to_bbox(mask_opened)

        # Draw: opened mask silhouette (cyan fill)
        draw_contour(img, mask_opened, (255, 255, 0), alpha=0.15)

        # Draw: opened bbox (cyan)
        if opened_bbox:
            draw_bbox(img, opened_bbox, (0, 255, 255),
                      f"mask+opening(k={k_size})", thickness=3)

        # Draw: raw mask bbox (green, thin)
        if raw_bbox:
            draw_bbox(img, raw_bbox, (0, 180, 0), "raw mask bbox", thickness=1)

        # 3. GroundingDINO (red)
        dets = detector.detect(img, args.caption, 0.25, 0.20)
        for d in dets:
            dd = d.as_dict()
            draw_bbox(img, dd["bbox_xyxy"], (0, 0, 255),
                      f"DINO ({dd['score']:.2f})", thickness=2)

        cv2.putText(img, f"img_{i:04d}", (img.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        panels.append(img)

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))
    grid = np.vstack([np.hstack(panels[r*cols:(r+1)*cols]) for r in range(rows)])

    leg = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    items = [
        (20, (0, 180, 0), "raw mask bbox"),
        (220, (0, 255, 255), "mask+opening bbox (tight)"),
        (520, (255, 255, 0), "opened silhouette"),
        (760, (0, 0, 255), "GroundingDINO"),
    ]
    for x, c, t in items:
        cv2.rectangle(leg, (x, 14), (x+22, 32), c, -1)
        cv2.putText(leg, t, (x+28, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    canvas = np.vstack([grid, leg])

    out = str(run_dir / "compare_mask_tight.png")
    cv2.imwrite(out, canvas)
    print(f"Saved: {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
