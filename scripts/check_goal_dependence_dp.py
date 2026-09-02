"""Goal-dependence probe for a trained Diffusion Policy.

Question: does the policy's action actually depend on the goal, or is it
collapsed (same action regardless of goal, output driven by sampling noise)?
Multiscale_bidir sampling was adopted to fix a measured collapse (ratio 0.67).

Metric: for held-out (val) start frames,
  goal_spread  = std of the (noise-averaged) first action ACROSS different goals
  noise_spread = std of the first action ACROSS noise draws, goal held fixed
  ratio        = goal_spread / noise_spread   (per start, then averaged)

ratio >> 1  -> action moves with the goal (goal-dependent, what we want)
ratio ~  1  -> goal and noise move it equally (weak)
ratio <  1  -> noise dominates (collapsed; the 0.67 symptom)

  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/check_goal_dependence_dp.py \
      --checkpoint runs/<run>/ckpt_best.pt --config configs/policy/diffusion_policy_dinov2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.policy.common.action_repr import ACTION_DIM
from src.policy.common.annotations import load_val_names
from src.policy.diffusion_policy.dataset import DiffusionPolicyDataset
from src.policy.diffusion_policy.model import DiffusionPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--repo_id", default="facebook/dinov2-large")
    ap.add_argument("--n_starts", type=int, default=24)
    ap.add_argument("--n_goals", type=int, default=10)
    ap.add_argument("--n_noise", type=int, default=6)
    ap.add_argument("--n_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["data"]
    keys = cfg["goal_score_keys"]
    dev = torch.device("cuda")
    dt = torch.bfloat16
    rng = np.random.default_rng(args.seed)

    # Held-out val split (object-disjoint) — cache hit, fast. Normalized goals +
    # images come out exactly as the model saw them in training.
    ds = DiffusionPolicyDataset(
        cfg["annotation_roots"], goal_score_keys=keys, chunk_size=cfg["chunk_size"],
        stride=cfg.get("val_stride", 4), sampling_scheme=cfg.get("sampling_scheme", "sliding_window"),
        offsets=cfg.get("offsets", [8, 16, 24]),
        goal_start_max_per_pair=int(cfg.get("goal_start_max_per_pair", 24)),
        target_resolution=tuple(cfg["target_resolution"]),
        val_pair_stride=int(cfg.get("val_pair_stride", 0)), val_split_level=cfg.get("val_split_level"),
        val_names=load_val_names(cfg.get("val_names")), split="val", goal_sampling="end",
        cache_dir=cfg.get("cache_dir"),
    )
    n = len(ds)
    print(f"val windows: {n}", flush=True)

    # A diverse goal pool + a set of start frames, drawn from distinct windows.
    pool_idx = rng.choice(n, size=min(args.n_goals, n), replace=False)
    goals = torch.stack([ds[i]["goal_vec"] for i in pool_idx]).float()  # (K, D) normalized
    start_idx = rng.choice(n, size=min(args.n_starts, n), replace=False)

    from transformers import AutoImageProcessor, AutoModel
    bb = AutoModel.from_pretrained(args.repo_id, torch_dtype=dt)
    pr = AutoImageProcessor.from_pretrained(args.repo_id)
    policy = DiffusionPolicy(bb, goal_dim=len(keys), chunk_size=cfg["chunk_size"], processor=pr).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["policy_state"], strict=False)
    print(f"DP loaded iter {ck.get('iteration')}", flush=True)

    K, M = len(goals), args.n_noise
    goal_sp, noise_sp = [], []          # per-start scalar spreads
    per_dim_goal, per_dim_noise = [], []
    for s in start_idx:
        img = ds[int(s)]["state_image"]                       # (3,H,W)
        # batch = K goals x M noise draws of the SAME start image
        imgs = img.unsqueeze(0).repeat(K * M, 1, 1, 1)
        g = goals.repeat_interleave(M, dim=0)                 # (K*M, D)
        batch = {"state_image": imgs, "goal_vec": g,
                 "action_chunk": torch.zeros(K * M, cfg["chunk_size"], ACTION_DIM)}
        with torch.no_grad(), torch.amp.autocast(dev.type, dtype=dt):
            oi, gg, _ = policy.prepare_inputs(batch, dev, dt)
            a0 = policy.sample(oi, gg, n_steps=args.n_steps).pred_action_chunk[:, 0, :]
        a0 = a0.float().cpu().numpy().reshape(K, M, ACTION_DIM)   # first action
        # goal spread: average out noise per goal, then std across goals
        gd = a0.mean(axis=1).std(axis=0)                      # (D,)
        # noise spread: std across noise per goal, then average across goals
        nd = a0.std(axis=1).mean(axis=0)                      # (D,)
        goal_sp.append(gd.mean()); noise_sp.append(nd.mean())
        per_dim_goal.append(gd); per_dim_noise.append(nd)

    gmean, nmean = float(np.mean(goal_sp)), float(np.mean(noise_sp))
    ratio = gmean / max(nmean, 1e-9)
    pdg, pdn = np.mean(per_dim_goal, axis=0), np.mean(per_dim_noise, axis=0)
    dim_ratio = pdg / np.maximum(pdn, 1e-9)
    labels = ["dR", "dU", "dF", "dYaw", "dPitch", "shoot"][:ACTION_DIM]

    print(f"\nstarts={len(start_idx)} goals={K} noise={M} steps={args.n_steps}")
    print(f"goal_spread  (action moves w/ goal):  {gmean:.4f}")
    print(f"noise_spread (action moves w/ noise): {nmean:.4f}")
    print(f"RATIO goal/noise = {ratio:.3f}   "
          f"[{'GOAL-DEPENDENT' if ratio > 2 else 'WEAK' if ratio > 1 else 'COLLAPSED'}]  (want > 2)")
    print("per-dim ratio: " + "  ".join(f"{l}={r:.2f}" for l, r in zip(labels, dim_ratio)))


if __name__ == "__main__":
    main()
