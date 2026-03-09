from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.scoring import flatten_scores_for_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations_path)
    image_root = Path(args.image_root) if args.image_root is not None else annotations_path.parent
    output_path = Path(args.output_path) if args.output_path is not None else annotations_path

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    for item in annotations:
        image_rel = str(item["image"])
        image_path = image_root / image_rel
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        score_fields = flatten_scores_for_annotation(
            annotation=item,
            image_width=image_width,
            image_height=image_height,
        )
        item.update(score_fields)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"saved scored annotations: {output_path}")


if __name__ == "__main__":
    main()
