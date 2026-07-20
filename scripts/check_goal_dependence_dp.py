"""Goal-dependence diagnostic for the DiffusionPolicy baseline (dedicated action head).

Same metric as check_goal_dependence.py but for the DINOv2 + 1D-U-Net DDPM policy: a
FIXED state, several goals x noise seeds, comparing between-goal vs within-goal action
spread. This is the PREMISE CHECK — does a dedicated (non-tiled) action diffusion head
learn goal-dependent actions on the multiscale data? (If yes, the Cosmos tiled-action-
latent is the culprit; if no, the problem is more fundamental than the architecture.)

  PYTHONPATH=. .venv/bin/python scripts/check_goal_dependence_dp.py \
      --checkpoint runs/<ts>_diffusion_policy_dinov2/ckpt_last.pt \
      --data-json data/trajectories/<placement>/data.json --n-steps 16 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.policy.common.annotations import load_annotation
from src.policy.common.goal_space import goal_keys, goal_vector, normalize_goal
from src.policy.diffusion_policy.dataset import _load_image_as_tensor
from src.policy.diffusion_policy.model import DiffusionPolicy


def load_dp(ckpt_path, repo_id, chunk_size, goal_dim, device, dtype):
    from transformers import AutoImageProcessor, AutoModel

    backbone = AutoModel.from_pretrained(repo_id, torch_dtype=dtype)
    processor = AutoImageProcessor.from_pretrained(repo_id)
    policy = DiffusionPolicy(
        backbone, goal_dim=goal_dim, chunk_size=chunk_size, processor=processor).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ckpt["policy_state"], strict=False)
    head_missing = [k for k in missing if not k.startswith("backbone.")]
    if head_missing:
        raise SystemExit(f"checkpoint HEAD mismatch (missing {head_missing[:6]}) — wrong chunk_size/goal_dim?")
    return policy, int(ckpt.get("iteration", -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-json", required=True, type=Path)
    ap.add_argument("--repo-id", default="facebook/dinov2-large")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--goal-frames", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--n-steps", nargs="+", type=int, default=[16])
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    keys = goal_keys()   # all 8 V5 keys (the training config uses all 8)
    policy, iteration = load_dp(args.checkpoint, args.repo_id, args.chunk_size, len(keys), device, dtype)
    print(f"loaded DiffusionPolicy iter={iteration} chunk_size={args.chunk_size} keys={keys}")

    doc = load_annotation(args.data_json)
    recs = (doc.get("render_records") or [[]])[args.pair]
    recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
    placement_dir = args.data_json.parent

    def frame_image(j: int) -> torch.Tensor:
        rr = recs_by_idx.get(j, {})
        rel = rr.get("path_rel", f"renders/pair_{args.pair:02d}_frame_{j:02d}.jpg")
        return _load_image_as_tensor(placement_dir / rel, tuple(args.resolution))

    state = frame_image(args.start_frame).unsqueeze(0)
    dummy = {"state_image": state, "goal_vec": torch.zeros(1, len(keys)),
             "action_chunk": torch.zeros(1, args.chunk_size, 5)}
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        obs_inputs, _, _ = policy.prepare_inputs(dummy, device, dtype)

    goals = []
    for gf in args.goal_frames:
        g = goal_vector(recs_by_idx.get(gf, {}).get("scores", {}), keys)
        if not np.isfinite(g).all():
            print(f"  skip goal frame {gf}: non-finite profile")
            continue
        goals.append((gf, normalize_goal(g, keys)))
    if len(goals) < 2:
        raise SystemExit("need >= 2 valid goal frames")

    for nsteps in args.n_steps:
        per_goal: dict[int, np.ndarray] = {}
        for gf, gnorm in goals:
            gt = torch.from_numpy(gnorm).unsqueeze(0).to(device, dtype=dtype)
            chunks = []
            for s in range(args.num_seeds):
                torch.manual_seed(1000 + s)
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = policy.sample(obs_inputs, gt, n_steps=nsteps, denormalize=False)
                chunks.append(out.pred_action_chunk.squeeze(0).float().cpu().numpy())
            per_goal[gf] = np.stack(chunks)

        within = float(np.mean([per_goal[gf].std(axis=0).mean() for gf, _ in goals]))
        goal_means = np.stack([per_goal[gf].mean(axis=0) for gf, _ in goals])
        between = float(goal_means.std(axis=0).mean())
        ratio = between / (within + 1e-9)
        verdict = "GOAL-DEPENDENT" if ratio > 2 else "WEAK / COLLAPSED"
        print(f"\n[n_steps={nsteps:>2}]  within(noise)={within:.4f}  between(signal)={between:.4f}  "
              f"RATIO={ratio:.2f}  ({verdict})")
        for gf, _ in goals:
            a = per_goal[gf].mean(axis=0)[0]
            print(f"    goal {gf:2d}: action[0] dx={a[0]:+.3f} dy={a[1]:+.3f} dz={a[2]:+.3f} "
                  f"dyaw={a[3]:+.3f} dpitch={a[4]:+.3f}")


if __name__ == "__main__":
    main()
