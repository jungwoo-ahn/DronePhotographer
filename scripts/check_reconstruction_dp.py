"""Train-set reconstruction check for the Diffusion Policy.

The lowest bar a policy must clear: on data it TRAINED on, when we sample an action
chunk, it should reconstruct the demonstrated action — i.e. applying the sampled
chunk from the start pose should land near the goal frame the chunk was built toward.
If it can't reconstruct train samples, closed-loop drift is a foregone conclusion.

Reports, over N train windows:
  * action MSE (sampled vs ground-truth, physical units, per dim)
  * reconstruction pose error: apply the SAMPLED chunk from the start pose, measure
    translation (cm) + rotation (deg) error to the goal frame — vs the same for the
    GT chunk (which should be ~0, a data/encode sanity check).

  CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python scripts/check_reconstruction_dp.py \
      --checkpoint runs/<run>/ckpt_best.pt --config configs/policy/diffusion_policy_dinov2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.policy.common.action_repr import POSE_DIM, apply_action_9d
from src.policy.common.annotations import load_val_names
from src.policy.common.dataset_base import BasePolicyDataset
from src.policy.common.goal_space import normalize_goal
from src.policy.diffusion_policy.dataset import _load_image_as_tensor
from src.policy.diffusion_policy.model import DiffusionPolicy


def _apply_chunk(start, chunk):
    """Apply an action chunk (chunk_size, >=5) from a start ViewRecord; return final pose."""
    p = np.asarray(start.camera_position, np.float32)
    f = np.asarray(start.camera_forward, np.float32)
    u = np.asarray(start.camera_up, np.float32)
    for a in chunk:
        p, f, u = apply_action_9d(p, f, u, a[:POSE_DIM])
    return p, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--repo_id", default="facebook/dinov2-large")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--n_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="val = held-out scenes (generalization, comparable to the VLA val recon)")
    ap.add_argument("--shuffle_goals", action="store_true",
                    help="feed each sample a DIFFERENT window's goal — if reconstruction stays "
                         "good, the policy IGNORES the goal (predicts average motion, not goal-following)")
    args = ap.parse_args()
    _full = yaml.safe_load(args.config.read_text())
    cfg = _full["data"]
    mcfg = _full.get("model", {})   # must match training (down_dims etc.) or load_state_dict silently mismatches
    keys = cfg["goal_score_keys"]
    dev = torch.device("cuda"); dt = torch.bfloat16
    rng = np.random.default_rng(args.seed)
    res = tuple(cfg["target_resolution"])

    # split=train: can it fit what it trained on. split=val: held-out scenes (generalization).
    ds = BasePolicyDataset(
        cfg["annotation_roots"], goal_score_keys=keys, chunk_size=cfg["chunk_size"],
        stride=cfg.get("stride", 1), sampling_scheme=cfg.get("sampling_scheme", "goal_start"),
        goal_start_max_per_pair=int(cfg.get("goal_start_max_per_pair", 24)),
        val_split_level=cfg.get("val_split_level"), val_names=load_val_names(cfg.get("val_names")),
        split=args.split, cache_dir=cfg.get("cache_dir"),
    )
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    print(f"{args.split} windows: {len(ds)}  sampling {len(idx)}", flush=True)

    from transformers import AutoImageProcessor, AutoModel
    bb = AutoModel.from_pretrained(args.repo_id, torch_dtype=dt)
    pr = AutoImageProcessor.from_pretrained(args.repo_id)
    policy = DiffusionPolicy(
        bb, goal_dim=len(keys), chunk_size=cfg["chunk_size"], processor=pr,
        goal_embed_dim=mcfg.get("goal_embed_dim", 128),
        down_dims=tuple(mcfg.get("down_dims", [128, 256, 512])),
        diffusion_step_embed_dim=mcfg.get("diffusion_step_embed_dim", 128),
        num_train_timesteps=mcfg.get("num_train_timesteps", 100),
        beta_schedule=mcfg.get("beta_schedule", "squaredcos_cap_v2"),
    ).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ck["policy_state"], strict=False)
    _crit = [k for k in (list(missing) + list(unexpected)) if "backbone" not in k]
    if _crit:
        raise SystemExit(f"checkpoint/model MISMATCH (non-backbone): {_crit[:6]} ... — config down_dims wrong?")
    print(f"DP loaded iter {ck.get('iteration')}", flush=True)

    from src.policy.diffusion_policy.dataset import build_obs_inputs
    mses, rec_cm, rec_deg, gt_cm = [], [], [], []
    # goal-USE probe: optionally feed each sample another window's goal (permuted).
    _goals = [normalize_goal(ds[int(i)].goal_vec, ds.goal_keys) for i in idx]
    _perm = (np.random.default_rng(args.seed + 1).permutation(len(idx))
             if args.shuffle_goals else np.arange(len(idx)))
    if args.shuffle_goals:
        print("SHUFFLED goals: each sample gets a DIFFERENT window's goal", flush=True)
    for j, i in enumerate(idx):
        s = ds[int(i)]
        img = _load_image_as_tensor(Path(s.start.image), res).unsqueeze(0)
        g = torch.from_numpy(_goals[_perm[j]]).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast(dev.type, dtype=dt):
            oi = build_obs_inputs(pr, img); oi = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in oi.items()}
            a_hat = policy.sample(oi, g.to(dev, dt), n_steps=args.n_steps).pred_action_chunk.squeeze(0).float().cpu().numpy()
        a_gt = s.action_chunk                                   # physical GT (chunk, 6)
        mses.append(np.mean((a_hat[:, :POSE_DIM] - a_gt[:, :POSE_DIM]) ** 2, axis=0))
        # The chunk walks to window.END (chunk_size steps, clamped) — NOT goal_frame,
        # which for far goals sits beyond the chunk. Reconstruct against `end`.
        gp = np.asarray(s.end.camera_position, np.float32); gf = np.asarray(s.end.camera_forward, np.float32)
        ph, fh = _apply_chunk(s.start, a_hat)
        pg, fg = _apply_chunk(s.start, a_gt)
        rec_cm.append(np.linalg.norm(ph - gp) * 100)
        rec_deg.append(np.degrees(np.arccos(np.clip(np.dot(fh / (np.linalg.norm(fh) + 1e-9), gf / (np.linalg.norm(gf) + 1e-9)), -1, 1))))
        gt_cm.append(np.linalg.norm(pg - gp) * 100)

    pd = np.mean(mses, axis=0)
    labels = ["tx","ty","tz","r0","r1","r2","r3","r4","r5"][:POSE_DIM]
    print(f"\n=== reconstruction on {len(idx)} TRAIN windows ===")
    print("action MSE (physical) per-dim: " + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, pd)) + f"  | mean={pd.mean():.4f}")
    print(f"reconstruct SAMPLED chunk -> end:  {np.mean(rec_cm):6.1f} cm  {np.mean(rec_deg):5.1f} deg  (median {np.median(rec_cm):.1f} cm)")
    print(f"reconstruct GT chunk -> end (sanity, MUST be ~0): {np.mean(gt_cm):.3f} cm")
    good = np.mean(np.array(rec_cm) < 20)
    print(f"windows reconstructed within 20cm: {100*good:.0f}%")


if __name__ == "__main__":
    main()
