"""Dump denoising samples (world/action/value + GT) from a Cosmos-policy checkpoint.

Standalone counterpart to the in-training viz hook: rebuilds the policy faithfully
(via rollout_eval.load_policy), draws a batch from a placement, and writes
`[state | pred-next-world | gt-next-world]` grids + a metrics.json to `--out`.

  PYTHONPATH=. .venv/bin/python scripts/log_cosmos_samples.py \
      --checkpoint runs/<ts>_cosmos_2b_goalnorm/ckpt_last.pt \
      --data-root data/trajectories/<placement> --out /tmp/samples_probe --n-samples 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_eval import load_policy  # faithful architecture rebuild + weight load

from src.policy.cosmos.dataset import CosmosDroneDataset
from src.policy.cosmos.sample_logging import log_denoise_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path, help="a placement dir (holds data.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    print(f"loaded policy iter={iteration} chunk_size={chunk_size}")

    ds = CosmosDroneDataset(
        [args.data_root], goal_score_keys=keys, chunk_size=chunk_size,
        sampling_scheme="multiscale_bidir", offsets=(8, 16, 24),
        target_resolution=tuple(args.resolution), split="train",
    )
    batch = next(iter(DataLoader(ds, batch_size=args.n_samples, shuffle=False)))
    summary = log_denoise_samples(
        policy, vae, batch, args.out, device=device, dtype=dtype,
        n_steps=args.n_steps, max_samples=args.n_samples,
    )
    print(f"wrote {args.out}  action_rmse_mean={summary['action_rmse_mean']}  "
          f"world_decode_error={summary['world_decode_error']}")
    for s in summary["samples"]:
        print(f"  sample{s['index']}: action_rmse={s['action_rmse']:.4f}"
              + (f" value_mae={s['value_mae']:.4f}" if 'value_mae' in s else "")
              + (f"  img={s.get('image', s.get('image_error'))}"))


if __name__ == "__main__":
    main()
