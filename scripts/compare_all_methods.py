"""Visualize 4 bbox methods: old AABB, vertex AABB, convex hull, GroundingDINO.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_all_methods.py \
        --run_dir outputs/DogWalk_v3_compare_260318_084442
"""
import argparse, json, sys
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
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hull(img, pts, color, label, alpha=0.1):
    arr = np.array(pts, dtype=np.float32)
    h, w = img.shape[:2]
    arr[:, 0] = np.clip(arr[:, 0], -w, 2*w)
    arr[:, 1] = np.clip(arr[:, 1], -h, 2*h)
    try:
        hull = ConvexHull(arr)
    except Exception:
        return
    hp = arr[hull.vertices].astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [hp], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [hp], True, color, 2)
    mx, my = hp[:, 0].min(), hp[:, 1].min()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    ty = max(my - 6, th + 4); tx = max(mx, 0)
    cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (tx + 2, ty - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--caption", default="a snowman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    annotations = json.loads((run_dir / "annotations.json").read_text())
    tight = json.loads((run_dir / "tight_test.json").read_text())

    detector = GroundingDINODetector(
        model_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=args.device,
    )

    panels = []
    for ann, t in zip(annotations, tight):
        img = cv2.imread(str(run_dir / ann["image"]))
        if img is None:
            continue

        # 1. Old AABB (green, thin)
        if ann.get("bbox_2d"):
            draw_bbox(img, ann["bbox_2d"], (0, 200, 0), "bound_box AABB", thickness=1)

        # 2. Vertex AABB (yellow)
        if t.get("bbox_tight"):
            draw_bbox(img, t["bbox_tight"], (0, 255, 255), "vertex AABB", thickness=2)

        # 3. Convex hull (cyan, filled)
        if t.get("hull_pts") and len(t["hull_pts"]) >= 3:
            draw_hull(img, t["hull_pts"], (255, 200, 0), "convex hull", alpha=0.10)

        # 4. GroundingDINO (red)
        dets = detector.detect(img, args.caption, 0.25, 0.20)
        for d in dets:
            dd = d.as_dict()
            draw_bbox(img, dd["bbox_xyxy"], (0, 0, 255), f"DINO ({dd['score']:.2f})", thickness=2)

        idx = Path(ann["image"]).stem
        cv2.putText(img, idx, (img.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        panels.append(img)

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    while len(panels) < rows * cols:
        panels.append(np.zeros_like(panels[0]))
    grid = np.vstack([np.hstack(panels[r*cols:(r+1)*cols]) for r in range(rows)])

    # Legend
    leg = np.zeros((45, grid.shape[1], 3), dtype=np.uint8)
    items = [
        (20, (0, 200, 0), "bound_box AABB (old)"),
        (280, (0, 255, 255), "vertex AABB (tight)"),
        (540, (255, 200, 0), "convex hull"),
        (730, (0, 0, 255), "GroundingDINO"),
    ]
    for x, c, txt in items:
        cv2.rectangle(leg, (x, 12), (x+25, 30), c, -1)
        cv2.putText(leg, txt, (x+32, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    canvas = np.vstack([grid, leg])

    out = str(run_dir / "compare_all_methods.png")
    cv2.imwrite(out, canvas)
    print(f"Saved: {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
