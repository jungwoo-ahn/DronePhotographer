"""Train-set reconstruction check for the VLA (π0-style: Qwen3-VL + flow ActionExpert, NL-text goal).

Same bar/metrics as scripts/check_reconstruction_dp.py: on TRAIN windows, sample the
action chunk and check it reconstructs the demonstrated chunk (lands near window.end).

  CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python scripts/check_reconstruction_vla.py \
      --checkpoint runs/<run>/ckpt_best.pt --config configs/policy/vla_qwen3_2b.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.policy.common.action_repr import ACTION_DIM, POSE_DIM, apply_action_9d
from src.policy.common.annotations import load_val_names
from src.policy.common.dataset_base import BasePolicyDataset
from src.policy.common.flow import FlowConfig
from src.policy.vla.dataset import _load_image_as_tensor
from src.policy.vla.model import VLAActionPolicy


def _apply(start, chunk):
    p = np.asarray(start.camera_position, np.float32); f = np.asarray(start.camera_forward, np.float32)
    u = np.asarray(start.camera_up, np.float32)
    for a in chunk:
        p, f, u = apply_action_9d(p, f, u, a[:POSE_DIM])
    return p, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    d = cfg["data"]; keys = d["goal_score_keys"]
    dev = torch.device("cuda"); dt = getattr(torch, cfg["trainer"]["dtype"])
    rng = np.random.default_rng(args.seed); res = tuple(d["target_resolution"])

    ds = BasePolicyDataset(
        d["annotation_roots"], goal_score_keys=keys, chunk_size=d["chunk_size"],
        sampling_scheme=d.get("sampling_scheme", "goal_start"),
        goal_start_max_per_pair=int(d.get("goal_start_max_per_pair", 24)),
        val_split_level=d.get("val_split_level"), val_names=load_val_names(d.get("val_names")),
        split="train", cache_dir=d.get("cache_dir"))
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    print(f"train windows: {len(ds)}  sampling {len(idx)}", flush=True)

    from transformers import AutoProcessor, Qwen3VLModel
    bb = Qwen3VLModel.from_pretrained(cfg["backbone"]["repo_id"], torch_dtype=dt,
                                      attn_implementation=cfg["backbone"].get("attn_implementation", "sdpa"))
    pk = {}
    if cfg["backbone"].get("max_pixels"): pk["max_pixels"] = int(cfg["backbone"]["max_pixels"])
    if cfg["backbone"].get("min_pixels"): pk["min_pixels"] = int(cfg["backbone"]["min_pixels"])
    proc = AutoProcessor.from_pretrained(cfg["backbone"]["repo_id"], **pk)
    flow = FlowConfig(**{k: v for k, v in cfg.get("flow", {}).items() if k in FlowConfig.__dataclass_fields__})
    policy = VLAActionPolicy(bb, goal_dim=len(keys), n_goal_tokens=cfg["backbone"]["n_goal_tokens"],
        chunk_size=d["chunk_size"], expert_dim=cfg["expert"]["dim"], expert_depth=cfg["expert"]["depth"],
        expert_heads=cfg["expert"]["heads"], expert_type=cfg["expert"].get("type", "mlp"),
        freeze_backbone=cfg["backbone"]["freeze_backbone"],
        flow_config=flow, processor=proc, goal_conditioning=cfg["backbone"].get("goal_conditioning", "text")).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ck["policy_state"], strict=False)
    _crit = [k for k in (list(missing) + list(unexpected)) if "backbone" not in k]
    if _crit:
        raise SystemExit(f"checkpoint/model MISMATCH (non-backbone): {_crit[:6]} ... — expert type/dims wrong?")
    print(f"VLA loaded iter {ck.get('iteration')}", flush=True)

    mses, rec_cm, rec_deg, gt_cm = [], [], [], []
    for i in idx:
        s = ds[int(i)]
        img = _load_image_as_tensor(Path(s.start.image), res)
        batch = {"state_image": img.unsqueeze(0),
                 "goal_raw": torch.from_numpy(np.asarray(s.goal_vec, np.float32)).unsqueeze(0),
                 "goal_vec": torch.zeros(1, len(keys)),
                 "action_chunk": torch.zeros(1, d["chunk_size"], ACTION_DIM),
                 "meta": [{"object": s.goal.object}]}
        with torch.no_grad(), torch.amp.autocast(dev.type, dtype=dt):
            vlm, goal, _ = policy.prepare_inputs(batch, dev, dt)
            a_hat = policy.sample(vlm, goal, n_steps=args.n_steps).pred_action_chunk.squeeze(0).float().cpu().numpy()
        a_gt = s.action_chunk
        mses.append(np.mean((a_hat[:, :POSE_DIM] - a_gt[:, :POSE_DIM]) ** 2, axis=0))
        gp = np.asarray(s.end.camera_position, np.float32); gf = np.asarray(s.end.camera_forward, np.float32)
        ph, fh = _apply(s.start, a_hat); pg, _ = _apply(s.start, a_gt)
        rec_cm.append(np.linalg.norm(ph - gp) * 100)
        rec_deg.append(np.degrees(np.arccos(np.clip(np.dot(fh/(np.linalg.norm(fh)+1e-9), gf/(np.linalg.norm(gf)+1e-9)), -1, 1))))
        gt_cm.append(np.linalg.norm(pg - gp) * 100)

    pd = np.mean(mses, axis=0); labels = ["tx","ty","tz","r0","r1","r2","r3","r4","r5"][:POSE_DIM]
    print(f"\n=== VLA reconstruction on {len(idx)} TRAIN windows ===")
    print("action MSE per-dim: " + "  ".join(f"{l}={v:.4f}" for l,v in zip(labels,pd)) + f"  | mean={pd.mean():.4f}")
    print(f"sampled -> end: {np.mean(rec_cm):6.1f} cm  {np.mean(rec_deg):5.1f} deg  (median {np.median(rec_cm):.1f} cm)")
    print(f"GT -> end (sanity ~0): {np.mean(gt_cm):.3f} cm")
    print(f"within 20cm: {100*np.mean(np.array(rec_cm)<20):.0f}%")


if __name__ == "__main__":
    main()
