"""Denoising-sample logging for the Cosmos world-action policy.

During training we periodically run the Euler sampler on a FIXED validation batch and
dump, per sample, into `<run>/samples/iter<NNNNNN>/`:

  - `sampleNN.png` — a horizontal grid `[ state | predicted next-world | GT next-world ]`.
    The predicted next-world is the model's world-head latent (position T_img-1) decoded
    back to an image: since world + action are sampled jointly, this frame IS "the world
    the predicted action leads to" in the model's imagination. GT is the real next frame.
  - `metrics.json` — per sample: the goal vector, predicted vs GT action chunk, predicted
    vs GT per-step value (both in physical units), and the action/value errors.

Pure inference + PIL; wrapped in try/except by the caller so a viz failure never kills a
run. Also runnable standalone on any checkpoint (see `scripts/log_cosmos_samples.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1, 1] -> (H, W, 3) uint8."""
    x = img_chw.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def _hgrid(*panels: np.ndarray, pad: int = 4) -> "Image.Image":
    """Concatenate equal-height (H, W, 3) uint8 panels horizontally with white gutters."""
    from PIL import Image

    h = max(p.shape[0] for p in panels)
    strips = []
    for p in panels:
        if p.shape[0] != h:  # pad shorter panels (shapes match in practice)
            p = np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0)), constant_values=255)
        strips.append(p)
        strips.append(np.full((h, pad, 3), 255, dtype=np.uint8))
    grid = np.concatenate(strips[:-1], axis=1)
    return Image.fromarray(grid)


@torch.no_grad()
def log_denoise_samples(
    policy,
    vae,
    batch: dict,
    out_dir: str | Path,
    *,
    device,
    dtype,
    n_steps: int = 16,
    max_samples: int = 4,
) -> dict:
    """Sample world/action/value on `batch`, decode the world frames, and dump images + json.

    Returns a small summary dict (also written as metrics.json). Never raises on a
    per-sample decode error — it records the error and continues.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = policy.training
    policy.eval()

    state = batch["state_image"].to(device, dtype=dtype)
    nxt = batch["next_state_image"].to(device, dtype=dtype)
    goal = batch["goal_vec"].to(device, dtype=dtype)
    gt_action = batch["action_chunk"].to(device, dtype=dtype)
    gt_value = batch["value_target"].to(device, dtype=dtype)
    b = min(max_samples, state.shape[0])

    with torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        image_latent = vae.encode_pair_frames(state, nxt)          # (B, 16, 2, h, w)
        pred = policy.sample(image_latent=image_latent, goal_vec=goal, n_steps=n_steps, denormalize=False)
        t_img = image_latent.shape[2]
        world_idx = t_img - 1                                       # last image frame = predicted next-world
        # Decode the model's predicted next-world latent back to an image.
        try:
            pred_world = vae.decode(pred.pred_latents[:, :, world_idx:world_idx + 1])[:, :, 0]  # (B,3,H,W)
        except Exception as e:                                      # decode is the only fragile step
            pred_world = None
            decode_err = repr(e)
        else:
            decode_err = None

    a_scale = policy.action_scale.detach().float().cpu().numpy()    # (5,) physical units
    v_scale = float(policy.value_scale)
    pred_a = pred.pred_action_chunk.detach().float().cpu().numpy()  # (B, chunk, 5) normalized
    gt_a = gt_action.detach().float().cpu().numpy()
    pred_v = pred.pred_value.detach().float().cpu().numpy() if pred.pred_value is not None else None
    gt_v = gt_value.detach().float().cpu().numpy()

    samples = []
    for i in range(b):
        rec = {"index": i, "goal_vec": goal[i].detach().float().cpu().numpy().round(3).tolist()}
        # action (physical units) — pred vs GT + per-chunk L2
        pa, ga = pred_a[i] * a_scale, gt_a[i] * a_scale
        rec["action_pred"] = pa.round(4).tolist()
        rec["action_gt"] = ga.round(4).tolist()
        rec["action_rmse"] = float(np.sqrt(((pa - ga) ** 2).mean()))
        if pred_v is not None:
            pv, gv = pred_v[i] * v_scale, gt_v[i] * v_scale
            rec["value_pred"] = np.asarray(pv).round(4).tolist()
            rec["value_gt"] = np.asarray(gv).round(4).tolist()
            rec["value_mae"] = float(np.abs(np.asarray(pv) - np.asarray(gv)).mean())
        # image grid [ state | pred next-world | gt next-world ]
        if pred_world is not None:
            try:
                grid = _hgrid(_chw_to_uint8(state[i]), _chw_to_uint8(pred_world[i]), _chw_to_uint8(nxt[i]))
                grid.save(out_dir / f"sample{i:02d}.png")
                rec["image"] = f"sample{i:02d}.png"
            except Exception as e:
                rec["image_error"] = repr(e)
        samples.append(rec)

    summary = {
        "n_steps": n_steps,
        "world_decode_error": decode_err,
        "grid_layout": "state | pred_next_world | gt_next_world",
        "action_rmse_mean": float(np.mean([s["action_rmse"] for s in samples])) if samples else None,
        "samples": samples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    if was_training:
        policy.train()
    return summary


__all__ = ["log_denoise_samples"]
