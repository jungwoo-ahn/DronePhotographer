from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drones.blender_drone import BlenderDrone


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one exact camera pose in Blender.")
    parser.add_argument("--run_info_path", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--sample_initial_pose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--position", nargs=3, type=float)
    parser.add_argument("--forward", nargs=3, type=float)
    parser.add_argument("--up", nargs=3, type=float)
    split_index = argv.index("--") + 1 if "--" in argv else len(argv)
    return parser.parse_args(argv[split_index:])


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    drone = BlenderDrone.from_run_info(args.run_info_path)

    if args.sample_initial_pose:
        pose = drone.sample_initial_pose(seed=args.seed)
    else:
        if args.position is None or args.forward is None or args.up is None:
            raise ValueError("position, forward, and up are required unless --sample_initial_pose is used")
        pose = drone.set_pose(
            position=args.position,
            forward=args.forward,
            up=args.up,
        )

    output_image = drone.render_rgb(args.output_image)
    payload = {
        "run_info_path": str(Path(args.run_info_path).resolve()),
        "image_path": str(output_image),
        "pose": pose,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raw_argv = getattr(bpy.app, "argv", None)
    if raw_argv is None:
        raw_argv = sys.argv
    raw_argv = list(raw_argv)
    if "--" not in raw_argv and raw_argv and raw_argv[0].endswith(".py"):
        raw_argv = raw_argv[1:]
    main(raw_argv)
