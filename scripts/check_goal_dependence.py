"""Goal-dependence diagnostic: does the policy's action actually depend on the goal?

For a FIXED state image, sample the policy under several DIFFERENT goals (each with several
noise seeds) and compare the spread of predicted actions ACROSS goals vs ACROSS seeds (same
goal). If actions collapsed to f(state), the between-goal spread ≈ the within-goal (noise)
spread → ratio ≈ 1. A healthy goal-conditioned policy has ratio ≫ 1.

Frame 0 of a v7 trajectory is the state; the goal frames (default the multiscale endpoints
8/16/24) supply the goals via their profiles — exactly the train-time construction, so this
reads out directly whether training fixed the "action ignores goal" collapse.

  PYTHONPATH=. .venv/bin/python scripts/check_goal_dependence.py \
      --checkpoint runs/<ts>_cosmos_2b/ckpt_last.pt \
      --data-json data/trajectories/<placement>/data.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.policy.common.annotations import _recover_clamped_goal, load_annotation
from src.policy.common.goal_space import goal_vector, normalize_goal
from src.policy.cosmos.dataset import _load_image_as_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_eval import load_policy   # faithful architecture rebuild + weight load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-json", required=True, type=Path)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--goal-frames", nargs="+", type=int, default=[8, 16, 24],
                    help="trajectory frames whose profiles are the goals (same start, different goals)")
    ap.add_argument("--num-seeds", type=int, default=8, help="noise draws per goal (within-goal spread)")
    ap.add_argument("--n-steps", nargs="+", type=int, default=[16],
                    help="Euler sampler steps; pass several to sweep (one ratio per value). "
                         "A flat sampled action_mse can be a low-step floor — bump this to test.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    ap.add_argument("--guidance-scale", type=float, default=1.0,
                    help="CFG scale; 1.0 = off. >1 amplifies the goal at inference.")
    ap.add_argument("--negative-mode", choices=["flip", "null"], default="flip",
                    help="CFG negative condition: point-symmetric flip goal, or null/unconditional")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    print(f"loaded policy iter={iteration} chunk_size={chunk_size} keys={keys}")

    doc = load_annotation(args.data_json)
    pair = doc["accepted_pairs"][args.pair]
    recs = (doc.get("render_records") or [[]])[args.pair]
    recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
    placement_dir = args.data_json.parent

    def frame_image(j: int) -> torch.Tensor:
        rr = recs_by_idx.get(j, {})
        rel = rr.get("path_rel", f"renders/pair_{args.pair:02d}_frame_{j:02d}.jpg")
        return _load_image_as_tensor(placement_dir / rel, tuple(args.resolution)).unsqueeze(0).to(device, dtype=dtype)

    state = frame_image(args.start_frame)
    with torch.no_grad():
        image_latent = vae.encode_pair_frames(state, state)

    goals = []
    for gf in args.goal_frames:
        rr = recs_by_idx.get(gf, {})
        # Rebuild the raw view + apply the clamp recovery so this probes the SAME
        # (recovered) goals training sees, not the baked near-plane sentinels.
        raw = dict(rr.get("scores") or {})
        raw["bbox_xyxy_full"] = rr.get("bbox_xyxy_full")
        raw["occupancy_clipped"] = rr.get("occupancy_clipped")
        _recover_clamped_goal(raw)
        g = goal_vector(raw, keys)
        if not np.isfinite(g).all():
            print(f"  skip goal frame {gf}: non-finite profile")
            continue
        goals.append((gf, normalize_goal(g, keys)))
    if len(goals) < 2:
        raise SystemExit("need >= 2 valid goal frames")

    # Sweep n_steps, reusing the loaded model + goals (model load is the expensive part).
    for nsteps in args.n_steps:
        per_goal: dict[int, np.ndarray] = {}   # gf -> (num_seeds, chunk, 5)
        for gf, gnorm in goals:
            gt = torch.from_numpy(gnorm).unsqueeze(0).to(device, dtype=dtype)
            chunks = []
            for s in range(args.num_seeds):
                torch.manual_seed(1000 + s)
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = policy.sample(image_latent=image_latent, goal_vec=gt, n_steps=nsteps,
                                        guidance_scale=args.guidance_scale, negative_mode=args.negative_mode)
                chunks.append(out.pred_action_chunk.squeeze(0).float().cpu().numpy())
            per_goal[gf] = np.stack(chunks)

        within = float(np.mean([per_goal[gf].std(axis=0).mean() for gf, _ in goals]))       # noise
        goal_means = np.stack([per_goal[gf].mean(axis=0) for gf, _ in goals])                # (G, chunk, 5)
        between = float(goal_means.std(axis=0).mean())                                       # signal
        ratio = between / (within + 1e-9)

        verdict = "GOAL-DEPENDENT" if ratio > 2 else "WEAK / COLLAPSED"
        cfg_tag = "" if args.guidance_scale == 1.0 else f"  [CFG s={args.guidance_scale} neg={args.negative_mode}]"
        print(f"\n[n_steps={nsteps:>2}]  within(noise)={within:.4f}  between(signal)={between:.4f}  "
              f"RATIO={ratio:.2f}  ({verdict}){cfg_tag}")
        for gf, _ in goals:
            a = per_goal[gf].mean(axis=0)[0]
            print(f"    goal {gf:2d}: action[0] dx={a[0]:+.3f} dy={a[1]:+.3f} dz={a[2]:+.3f} "
                  f"dyaw={a[3]:+.3f} dpitch={a[4]:+.3f}")


if __name__ == "__main__":
    main()
