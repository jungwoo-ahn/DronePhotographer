"""Train-set reconstruction check for the pretrained-VLA baselines (LeRobot pi05 / groot).

Same bar as scripts/check_reconstruction_dp.py: on TRAIN windows, sample the action chunk
from the policy and check that applying it from the start pose lands near the walk endpoint
(window.end) — reported in cm / deg, plus per-dim action MSE and a GT-chunk sanity (~0).

Poses come from OUR BasePolicyDataset windows (the LeRobot dataset stores no camera pose);
the policy input (start image + zero state + NL goal prompt) is built from the same window,
so the two are aligned per-sample.

  conda activate vla_groot   # (or vla)
  CUDA_VISIBLE_DEVICES=N HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    python scripts/check_reconstruction_lerobot.py \
      --policy pi05  --checkpoint runs/pi05_drone_interim/checkpoints/000500/pretrained_model
      --policy groot --checkpoint runs/groot_drone_interim/checkpoints/000500/pretrained_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.policy.common.action_repr import POSE_DIM, apply_action_9d
from src.policy.common.annotations import iter_goal_start_windows
from src.policy.common.dataset_base import _compute_action_chunk
from src.policy.common.goal_space import goal_vector
from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt

VIDEO_KEY = "observation.images.image"


def _load_chw(path: Path, size: int) -> torch.Tensor:
    """RGB float[0,1] CHW, resized square — matches LeRobotDataset image output."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)  # HWC -> CHW


def _apply(start, chunk):
    p = np.asarray(start.camera_position, np.float32)
    f = np.asarray(start.camera_forward, np.float32)
    u = np.asarray(start.camera_up, np.float32)
    for a in chunk:
        p, f, u = apply_action_9d(p, f, u, a[:POSE_DIM])
    return p, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["pi05", "groot"])
    ap.add_argument("--checkpoint", required=True, help="path to a checkpoint's pretrained_model dir")
    ap.add_argument("--roots", nargs="+", default=["data/trajectories_full"])
    ap.add_argument("--scenes", choices=["train", "val", "all"], default="all",
                    help="restrict recon windows by scene split (val_scenes.json): "
                         "val=held-out generalization, train=fit check, all=both")
    ap.add_argument("--val-scenes", default="configs/policy/val_scenes.json", dest="val_scenes")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--state_dim", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_goals", type=int, default=0,
                    help="goal-dependence probe: also predict with a MISMATCHED goal prompt on "
                         "the same start image. If shuffled recon ~= correct recon and |Δaction| ~= 0, "
                         "the policy IGNORES the goal (degenerate / not conditioning).")
    args = ap.parse_args()

    dev = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    # --- policy + processors from the checkpoint ---
    from lerobot.policies.factory import make_pre_post_processors
    if args.policy == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Pol
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy as Pol
    policy = Pol.from_pretrained(args.checkpoint).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, args.checkpoint)
    print(f"{args.policy} loaded from {args.checkpoint}", flush=True)

    # --- collect windows (poses + goal + start image), filtered by scene split ---
    import json
    import os
    val_scenes = frozenset()
    if args.scenes != "all":
        val_scenes = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
    def _keep(placement: str) -> bool:
        if args.scenes == "all":
            return True
        in_val = placement.split("__")[0] in val_scenes
        return in_val if args.scenes == "val" else (not in_val)
    windows = []
    for root in args.roots:
        for d in sorted(os.listdir(root)):
            if not _keep(d):
                continue
            p = Path(root) / d / "data.json"
            if not p.exists():
                continue
            try:
                for w in iter_goal_start_windows(p, chunk_size=args.chunk_size, max_per_pair=2):
                    windows.append(w)
            except Exception:
                continue
            if len(windows) > 5000:
                break
    idx = rng.choice(len(windows), size=min(args.n, len(windows)), replace=False)
    # precompute valid (window, correct-prompt); shuffle needs a mismatched-goal pool
    valid = []
    for i in idx:
        w = windows[int(i)]
        g = goal_vector(w.goal_frame.raw, NL_GOAL_KEYS)
        if not np.isfinite(g).all():
            continue
        valid.append((w, goal_prompt(g, NL_GOAL_KEYS, crop=w.goal_frame.raw)))
    print(f"windows: {len(windows)}  sampling {len(idx)}  valid {len(valid)}", flush=True)

    # near-derangement so each window is scored against a DIFFERENT window's goal
    perm = rng.permutation(len(valid))
    for j in range(len(perm)):
        if perm[j] == j and len(perm) > 1:
            k = (j + 1) % len(perm); perm[j], perm[k] = perm[k], perm[j]

    zero_state = torch.zeros(args.state_dim, dtype=torch.float32)

    def _predict(prompt, img):
        batch = {VIDEO_KEY: img, "observation.state": zero_state, "task": prompt}
        with torch.no_grad():
            return post(policy.predict_action_chunk(pre(batch))).squeeze(0).float().cpu().numpy()[:, :10]

    rec_cm, rec_deg, gt_cm, mses = [], [], [], []
    rec_cm_shuf, act_l1 = [], []
    for j, (w, prompt) in enumerate(valid):
        img = _load_chw(Path(w.keyframes[0].image), args.resize)
        a_hat = _predict(prompt, img)
        a_gt = _compute_action_chunk(w)                   # (chunk, 10)
        h = min(len(a_hat), len(a_gt))
        mses.append(np.mean((a_hat[:h, :POSE_DIM] - a_gt[:h, :POSE_DIM]) ** 2, axis=0))
        gp = np.asarray(w.keyframes[-1].camera_position, np.float32)
        gf = np.asarray(w.keyframes[-1].camera_forward, np.float32)
        ph, fh = _apply(w.keyframes[0], a_hat[:h])
        pg, _ = _apply(w.keyframes[0], a_gt[:h])
        rec_cm.append(np.linalg.norm(ph - gp) * 100)
        rec_deg.append(np.degrees(np.arccos(np.clip(
            np.dot(fh / (np.linalg.norm(fh) + 1e-9), gf / (np.linalg.norm(gf) + 1e-9)), -1, 1))))
        gt_cm.append(np.linalg.norm(pg - gp) * 100)
        if args.shuffle_goals:
            a_shuf = _predict(valid[perm[j]][1], img)     # SAME start image, WRONG goal
            hs = min(len(a_shuf), len(a_gt))
            ps, _ = _apply(w.keyframes[0], a_shuf[:hs])
            rec_cm_shuf.append(np.linalg.norm(ps - gp) * 100)
            act_l1.append(float(np.mean(np.abs(a_hat[:h, :POSE_DIM] - a_shuf[:h, :POSE_DIM]))))

    pd = np.mean(mses, axis=0)
    labels = ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5"][:POSE_DIM]
    print(f"\n=== {args.policy} reconstruction on {len(rec_cm)} {args.scenes.upper()} windows ===")
    print("action MSE per-dim: " + "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, pd)) + f"  | mean={pd.mean():.4f}")
    print(f"reconstruct SAMPLED chunk -> end: {np.mean(rec_cm):6.1f} cm  {np.mean(rec_deg):5.1f} deg  (median {np.median(rec_cm):.1f} cm)")
    print(f"reconstruct GT chunk -> end (sanity, MUST be ~0): {np.mean(gt_cm):.3f} cm")
    print(f"within 20cm: {100*np.mean(np.array(rec_cm) < 20):.0f}%")
    if args.shuffle_goals:
        # action-scale reference: mean |a_gt| over pose dims, so |Δaction| is interpretable
        ref = float(np.mean([np.mean(np.abs(_compute_action_chunk(w)[:, :POSE_DIM])) for w, _ in valid]))
        print(f"\n--- GOAL-DEPENDENCE PROBE (same start image, shuffled goal) ---")
        print(f"recon cm:  correct-goal {np.mean(rec_cm):6.1f}   shuffled-goal {np.mean(rec_cm_shuf):6.1f}   "
              f"(shuffled >> correct => USES goal)")
        print(f"mean |Δaction| correct-vs-shuffled: {np.mean(act_l1):.5f}   (vs mean|action|={ref:.5f}; "
              f"ratio {np.mean(act_l1)/max(ref,1e-9):.2f} — ~0 => IGNORES goal)")


if __name__ == "__main__":
    main()
