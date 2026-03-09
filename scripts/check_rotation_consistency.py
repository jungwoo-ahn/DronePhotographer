from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_qwen25.rotation_utils import (
    make_camera_rotation_from_forward_up,
    relative_rotation_matrix,
    relative_rotation_rotvec,
    rotation_quality,
    rotvec_to_rotation_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="outputs/DogWalk_260215_092109/annotations copy.json",
    )
    parser.add_argument("--max_views", type=int, default=5000)
    parser.add_argument("--pair_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--det_tol", type=float, default=1e-4)
    parser.add_argument("--orth_tol", type=float, default=1e-4)
    parser.add_argument("--recon_tol", type=float, default=1e-4)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _get_forward_up(item: dict) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(item.get("final_forward", item.get("base_forward")), dtype=np.float32)
    up = np.asarray(item.get("final_up", item.get("base_up")), dtype=np.float32)
    return forward, up


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    annotations_path = Path(args.annotations_path)
    with annotations_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = [item for item in raw if item.get("final_forward") or item.get("base_forward")]
    if not items:
        raise ValueError("no items with forward/up vectors found")

    if args.max_views > 0:
        items = items[: min(args.max_views, len(items))]

    rotations: list[np.ndarray] = []
    det_errors: list[float] = []
    orth_errors: list[float] = []

    for item in items:
        forward, up = _get_forward_up(item)
        rotation = make_camera_rotation_from_forward_up(forward, up)
        rotations.append(rotation)
        det_err, orth_err = rotation_quality(rotation)
        det_errors.append(det_err)
        orth_errors.append(orth_err)

    pair_count = min(args.pair_samples, len(rotations) * (len(rotations) - 1))
    recon_errors: list[float] = []

    if pair_count > 0:
        for _ in range(pair_count):
            i = rng.randrange(len(rotations))
            j = rng.randrange(len(rotations) - 1)
            if j >= i:
                j += 1

            fi, ui = _get_forward_up(items[i])
            fj, uj = _get_forward_up(items[j])

            rel_matrix = relative_rotation_matrix(fi, ui, fj, uj)
            rel_rotvec = relative_rotation_rotvec(fi, ui, fj, uj)
            rel_matrix_from_rotvec = rotvec_to_rotation_matrix(rel_rotvec)

            err = float(np.linalg.norm(rel_matrix - rel_matrix_from_rotvec, ord="fro"))
            recon_errors.append(err)

    report = {
        "annotations_path": str(annotations_path),
        "num_views_checked": len(rotations),
        "num_pairs_checked": len(recon_errors),
        "max_det_error": float(max(det_errors)),
        "max_orth_error": float(max(orth_errors)),
        "max_rel_recon_error": 0.0 if not recon_errors else float(max(recon_errors)),
        "mean_det_error": float(sum(det_errors) / len(det_errors)),
        "mean_orth_error": float(sum(orth_errors) / len(orth_errors)),
        "mean_rel_recon_error": 0.0 if not recon_errors else float(sum(recon_errors) / len(recon_errors)),
        "tolerances": {
            "det_tol": float(args.det_tol),
            "orth_tol": float(args.orth_tol),
            "recon_tol": float(args.recon_tol),
        },
    }

    passes = (
        report["max_det_error"] <= args.det_tol
        and report["max_orth_error"] <= args.orth_tol
        and report["max_rel_recon_error"] <= args.recon_tol
    )
    report["pass"] = bool(passes)

    print(json.dumps(report, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.strict and not passes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
