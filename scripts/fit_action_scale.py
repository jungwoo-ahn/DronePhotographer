"""Recompute the per-dimension action normalization scale (ACTION_SCALE).

The 5D action is computed from the camera poses in `accepted_pairs[].trajectory_32f`,
which exist as of Stage 1 — so this needs only trajectory data (no renders/scores).
Prints the per-dim p99 of |Δ| over all per-step actions; paste the result into
`src.policy.common.action_repr.ACTION_SCALE` (or load it at train time and store
it with the checkpoint).

Usage:
  python scripts/fit_action_scale.py --roots data/trajectories outputs/v7_stage1_sample
  python scripts/fit_action_scale.py --roots data/trajectories --percentile 99 --max-files 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.policy.common.action_repr import ACTION_DIM, ACTION_SCALE, encode_action_5d
from src.policy.common.annotations import list_annotation_files, load_annotation

_NAMES = ["dx(right)", "dy(up)", "dz(fwd)", "dyaw", "dpitch"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, type=Path)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-files", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = list_annotation_files(args.roots)
    if args.max_files:
        files = files[: args.max_files]

    actions: list[np.ndarray] = []
    for f in files:
        try:
            doc = load_annotation(f)
        except (OSError, ValueError):
            continue
        for pair in doc.get("accepted_pairs", []):
            traj = pair.get("trajectory_32f") or []
            for j in range(len(traj) - 1):
                a, b = traj[j], traj[j + 1]
                actions.append(encode_action_5d(
                    np.asarray(a["pos"], np.float32), np.asarray(a["forward"], np.float32), np.asarray(a["up"], np.float32),
                    np.asarray(b["pos"], np.float32), np.asarray(b["forward"], np.float32), np.asarray(b["up"], np.float32),
                ))
    if not actions:
        raise SystemExit("no trajectory actions found under the given roots")

    A = np.stack(actions)
    scale = np.percentile(np.abs(A), args.percentile, axis=0)
    print(f"files={len(files)}  actions={len(A)}  percentile={args.percentile}")
    print(f"{'dim':10s}{'p'+str(int(args.percentile)):>10s}{'current':>10s}{'ratio':>8s}")
    for i in range(ACTION_DIM):
        print(f"{_NAMES[i]:10s}{scale[i]:10.3f}{float(ACTION_SCALE[i]):10.3f}{scale[i]/float(ACTION_SCALE[i]):8.2f}")
    print("\nACTION_SCALE = np.array([" + ", ".join(f"{v:.3f}" for v in scale) + "], dtype=np.float32)")


if __name__ == "__main__":
    main()
