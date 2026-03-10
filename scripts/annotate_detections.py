from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detectors.detector import GroundingDINODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument(
        "--dino_config",
        type=str,
        default="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations_path)
    image_root = Path(args.image_root) if args.image_root is not None else annotations_path.parent
    output_path = Path(args.output_path) if args.output_path is not None else annotations_path

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    detector = GroundingDINODetector(
        model_config_path=args.dino_config,
        model_checkpoint_path=args.dino_weights,
        device=args.device,
    )

    total = len(annotations) if args.limit <= 0 else min(len(annotations), args.limit)
    for idx in tqdm(range(total), desc="detect"):
        item = annotations[idx]
        if (not args.overwrite) and item.get("detections"):
            continue
        image_path = image_root / str(item["image"])
        detections = detector.detect_file(
            image_path=image_path,
            caption=args.caption,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
        )
        item["prompt"] = str(args.caption)
        item["detections"] = [d.as_dict() for d in detections]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"saved annotations with detections: {output_path}")


if __name__ == "__main__":
    main()

"""
Example usage:


CUDA_VISIBLE_DEVICES=7 python scripts/annotate_detections.py \
  --annotations_path outputs/DogWalk_v2_10k_260309_101152/annotations.json \
  --image_root outputs/DogWalk_v2_10k_260309_101152 \
  --caption "a snowman" \
  --device cuda \
  --output_path outputs/DogWalk_v2_10k_260309_101152/annotations_detected.json

check visualization with:
conda activate drone

python scripts/visualize_detections.py \
  --annotations_path outputs/DogWalk_v2_10k_260309_101152/annotations_detected.json \
  --image_root outputs/DogWalk_v2_10k_260309_101152 \
  --output_dir outputs/DogWalk_v2_10k_260309_101152/detection_viz \
  --limit 20

"""
