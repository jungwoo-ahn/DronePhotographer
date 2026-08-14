"""Pre-build the window-index disk cache for a policy config, CPU-only, torch-free.

The cache is built entirely inside BasePolicyDataset.__init__ (numpy only); the
DP/VLA wrappers add torch solely for image loading at __getitem__, which the
cache doesn't touch. So we construct BasePolicyDataset directly with the exact
arguments the wrappers forward to it — same key inputs, same cache file — and
avoid needing torch here.

Run this first (holds no GPU, ~1h for the ~5M multiscale train split), then
launch training: it loads the index in seconds and claims a fresh GPU
immediately, with no long CPU-build window where the GPU looks free to others.

  python scripts/build_index_cache.py --config configs/policy/diffusion_policy_dinov2.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.policy.common.annotations import load_val_names
from src.policy.common.dataset_base import BasePolicyDataset


def _build(split: str, *, roots, stride, goal_sampling, max_samples, common) -> int:
    # Mirrors the wrappers' forward to BasePolicyDataset. filter_clamped_goals
    # is the wrappers' default (True) — the train scripts never override it.
    t = time.time()
    ds = BasePolicyDataset(
        roots,
        goal_score_keys=common["goal_score_keys"],
        chunk_size=common["chunk_size"],
        stride=stride,
        max_samples=max_samples,
        filter_clamped_goals=True,
        goal_sampling=goal_sampling,
        sampling_scheme=common["sampling_scheme"],
        offsets=common["offsets"],
        goal_start_max_per_pair=common["goal_start_max_per_pair"],
        goal_start_seed=common["goal_start_seed"],
        val_pair_stride=common["val_pair_stride"],
        val_split_level=common["val_split_level"],
        val_names=common["val_names"],
        split=split,
        cache_dir=common["cache_dir"],
    )
    print(f"[cache] {split:5s} split: {len(ds)} windows  ({time.time()-t:.0f}s)", flush=True)
    return len(ds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not cfg["data"].get("cache_dir"):
        raise SystemExit("config has no data.cache_dir — nothing to pre-build")

    roots = cfg["data"]["annotation_roots"]
    val_pair_stride = int(cfg["data"].get("val_pair_stride", 0))
    val_names = load_val_names(cfg["data"].get("val_names"))
    common = dict(
        goal_score_keys=cfg["data"]["goal_score_keys"],
        chunk_size=cfg["data"]["chunk_size"],
        sampling_scheme=cfg["data"].get("sampling_scheme", "sliding_window"),
        offsets=cfg["data"].get("offsets", [8, 16, 24]),
        goal_start_max_per_pair=int(cfg["data"].get("goal_start_max_per_pair", 24)),
        goal_start_seed=int(cfg["data"].get("goal_start_seed", 0)),
        val_pair_stride=val_pair_stride,
        val_split_level=cfg["data"].get("val_split_level", "scene"),
        val_names=val_names,
        cache_dir=cfg["data"].get("cache_dir"),
    )

    _build("train", roots=roots, stride=cfg["data"].get("stride", 1),
           goal_sampling=cfg["data"].get("goal_sampling", "uniform_future"),
           max_samples=cfg["data"].get("max_samples"), common=common)
    if val_pair_stride > 0 or val_names:
        _build("val", roots=roots, stride=cfg["data"].get("val_stride", 4),
               goal_sampling="end", max_samples=None, common=common)

    cd = Path(common["cache_dir"])
    files = sorted(cd.glob("win_index_*.pkl"))
    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"[cache] {len(files)} file(s), {total:.1f} MB in {cd}", flush=True)
    for p in files:
        print(f"        {p.name}  {p.stat().st_size/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
