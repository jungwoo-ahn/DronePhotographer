"""Recompute ACTION_SCALE (per-dim) and VALUE_SCALE from the training sampling scheme.

Actions and the per-step value are computed EXACTLY as the dataset builds them
(`_compute_action_chunk` / `_compute_value_sequence` over the same windows), so the
normalization matches what the model sees.

For the default `multiscale_bidir` scheme the actions include the STRIDED offset-16/24
chunks, which are 2-3x larger than single steps — so ACTION_SCALE MUST be refit for that
scheme, or the far-goal actions saturate at ±1 and lose all training signal. The value
p99 gives VALUE_SCALE (cosmos-policy-style [-1,1] value normalization).

Needs only Stage-1 trajectory data (poses + subject geometry); no renders/scores.

Usage:
  python scripts/fit_action_scale.py --roots data/trajectories
  python scripts/fit_action_scale.py --roots data/trajectories --sampling multiscale_bidir --offsets 8 16 24
  python scripts/fit_action_scale.py --roots data/trajectories --sampling sliding_window --chunk-size 8
  python scripts/fit_action_scale.py --roots data/trajectories --percentile 99 --max-files 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.policy.common.action_repr import ACTION_DIM, ACTION_SCALE
from src.policy.common.annotations import (
    iter_multiscale_windows,
    iter_windows,
    list_annotation_files,
)
from src.policy.common.dataset_base import _compute_action_chunk, _compute_value_sequence
from src.policy.common.reward import VALUE_SCALE

_NAMES = ["dx(right)", "dy(up)", "dz(fwd)", "dyaw", "dpitch"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, type=Path)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--sampling", choices=["multiscale_bidir", "sliding_window"],
                   default="multiscale_bidir", help="scheme to fit against (match the train config)")
    p.add_argument("--offsets", nargs="+", type=int, default=[8, 16, 24],
                   help="multiscale endpoint offsets (multiples of chunk-size)")
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="sliding_window stride")
    return p.parse_args()


def _windows(f: Path, args: argparse.Namespace):
    if args.sampling == "multiscale_bidir":
        return iter_multiscale_windows(f, chunk_size=args.chunk_size, offsets=tuple(args.offsets))
    return iter_windows(f, chunk_size=args.chunk_size, stride=args.stride)


def main() -> None:
    args = parse_args()
    files = list_annotation_files(args.roots)
    if args.max_files:
        files = files[: args.max_files]

    actions: list[np.ndarray] = []      # each (chunk_size, ACTION_DIM)
    values: list[np.ndarray] = []       # each (chunk_size,)
    n_windows = 0
    for f in files:
        try:
            for w in _windows(f, args):
                actions.append(_compute_action_chunk(w))
                values.append(_compute_value_sequence(w, w.end))   # goal = window endpoint
                n_windows += 1
        except (OSError, ValueError, KeyError):
            continue
    if not actions:
        raise SystemExit("no windows found under the given roots")

    A = np.concatenate([a.reshape(-1, ACTION_DIM) for a in actions], axis=0)   # (N·chunk, 5)
    V = np.concatenate([v.reshape(-1) for v in values], axis=0)                # (N·chunk,)
    a_scale = np.percentile(np.abs(A), args.percentile, axis=0)
    v_scale = float(np.percentile(np.abs(V), args.percentile))

    print(f"sampling={args.sampling}  files={len(files)}  windows={n_windows}  "
          f"per-step actions={len(A)}  percentile={args.percentile}")
    print(f"{'dim':10s}{'p' + str(int(args.percentile)):>10s}{'current':>10s}{'ratio':>8s}")
    for i in range(ACTION_DIM):
        print(f"{_NAMES[i]:10s}{a_scale[i]:10.3f}{float(ACTION_SCALE[i]):10.3f}"
              f"{a_scale[i] / float(ACTION_SCALE[i]):8.2f}")
    print(f"{'value':10s}{v_scale:10.3f}{float(VALUE_SCALE):10.3f}{v_scale / float(VALUE_SCALE):8.2f}")
    print("\n# paste into src/policy/common/action_repr.py")
    print("ACTION_SCALE = np.array([" + ", ".join(f"{v:.3f}" for v in a_scale) + "], dtype=np.float32)")
    print("# paste into src/policy/common/reward.py")
    print(f"VALUE_SCALE: float = {v_scale:.3f}")


if __name__ == "__main__":
    main()
