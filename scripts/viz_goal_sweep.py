"""Goal-sweep viz: for a FIXED state, show goal -> (predicted world, action, value) vs GT.

Reads out goal-dependence *legibly*: one fixed start frame, several multiscale goals
(+offsets, the exact train/diagnostic construction). Per goal we render a row

    [ predicted next-world | GT next-world | goal profile + pred-vs-GT action + value ]

so you can scan down and see EXACTLY which goal produced which action, next to the GT the
data actually has for that goal. A footer quantifies how much the predicted action[0]
varies across goals (the model's goal signal) vs how much the GT does (the data signal).

  PYTHONPATH=. .venv/bin/python scripts/viz_goal_sweep.py \
      --checkpoint runs/<ts>_cosmos_2b_goalnorm/ckpt_last.pt \
      --data-root data/trajectories/<placement> --out /tmp/goal_sweep.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_eval import load_policy

from src.policy.common.annotations import iter_multiscale_windows
from src.policy.common.dataset_base import _compute_action_chunk, _compute_value_sequence
from src.policy.common.goal_space import goal_vector
from src.policy.cosmos.dataset import _load_image_as_tensor

# raw-profile display: (label, unit, fmt) per goal key
PROFILE_FMT = [
    ("occupancy", "%", "{:.0f}"), ("body_in_frame", "%", "{:.0f}"),
    ("azimuth", "deg", "{:.0f}"), ("elevation", "deg", "{:+.0f}"),
    ("obj_center_x", "px", "{:.0f}"), ("obj_center_y", "px", "{:.0f}"),
    ("bbox_x_off", "px", "{:.0f}"), ("bbox_y_off", "px", "{:.0f}"),
]
ADIMS = ["R", "U", "F", "yaw", "pit"]   # right/up/fwd (m), yaw/pitch (rad)


def _img(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [-1,1] -> (H,W,3) float in [0,1] for imshow."""
    return (t.detach().float().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path, help="placement dir (holds data.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--offsets", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--seeds", type=int, default=4, help="noise draws averaged for the action (goal effect)")
    ap.add_argument("--n-steps", type=int, default=16)
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale; 1.0 = off")
    ap.add_argument("--negative-mode", choices=["flip", "null"], default="flip", help="CFG negative condition")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    res = tuple(args.resolution)
    dj = args.data_root / "data.json"

    # multiscale windows for this exact (pair, start frame), forward, sorted by offset
    wins = [w for w in iter_multiscale_windows(dj, chunk_size=chunk_size, offsets=tuple(args.offsets))
            if getattr(w, "direction", 1) == 1 and w.pair_idx == args.pair and w.start_frame_idx == args.start_frame]
    wins.sort(key=lambda w: w.end_frame_idx - w.start_frame_idx)
    if not wins:
        raise SystemExit(f"no forward windows for pair={args.pair} start={args.start_frame}")

    # state = start frame image (from the first keyframe of any window)
    state_rec = wins[0].keyframes[0]
    state = _load_image_as_tensor(Path(state_rec.image), res).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        image_latent = vae.encode_pair_frames(state, state)
    t_img = image_latent.shape[2]

    rows = []
    for w in wins:
        offset = w.end_frame_idx - w.start_frame_idx
        goal_raw = goal_vector(w.end.raw, keys)                       # raw profile (named units)
        from src.policy.common.goal_space import normalize_goal
        goal_norm = normalize_goal(goal_raw, keys)
        gt_action = _compute_action_chunk(w)                          # (chunk,5) physical
        gt_value = _compute_value_sequence(w, w.end, "cost_to_go", keys)  # (chunk,)
        gt_world = _load_image_as_tensor(Path(w.end.image), res)      # (3,H,W)

        gt = torch.from_numpy(goal_norm).unsqueeze(0).to(device, dtype=dtype)
        pred_actions, world0, val0 = [], None, None
        for s in range(args.seeds):
            torch.manual_seed(2000 + s)
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = policy.sample(image_latent=image_latent, goal_vec=gt, n_steps=args.n_steps,
                                    guidance_scale=args.guidance_scale, negative_mode=args.negative_mode)
            pred_actions.append(out.pred_action_chunk.squeeze(0).float().cpu().numpy())  # physical
            if s == 0:
                val0 = out.pred_value.squeeze(0).float().cpu().numpy() if out.pred_value is not None else None
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    world0 = vae.decode(out.pred_latents[:, :, t_img - 1:t_img])[:, :, 0][0]
        pred_action = np.mean(pred_actions, axis=0)                   # (chunk,5), mean over seeds = goal effect
        rows.append(dict(offset=offset, end=w.end_frame_idx, goal_raw=goal_raw,
                         pred_a=pred_action, gt_a=gt_action, pred_w=world0, gt_w=gt_world,
                         pred_v=val0, gt_v=gt_value))

    _render(rows, state[0], args, iteration)
    print(f"wrote {args.out}  ({len(rows)} goals, seed-avg action over {args.seeds} seeds)")


def _render(rows, state_img, args, iteration) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = len(rows)
    fig = plt.figure(figsize=(15, 2.7 * (G + 1)), dpi=110)
    gs = fig.add_gridspec(G + 1, 3, width_ratios=[1, 1, 1.05], hspace=0.12, wspace=0.05)

    # top row: fixed state
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(_img(state_img)); ax.set_title("STATE (fixed)", fontsize=11, weight="bold"); ax.axis("off")
    axh = fig.add_subplot(gs[0, 1:]); axh.axis("off")
    axh.text(0.0, 0.5,
             f"GOAL-SWEEP  ·  iter {iteration}  ·  {args.data_root.name}\n"
             f"pair {args.pair}, start frame {args.start_frame}  ·  same state, {G} goals (+{'/+'.join(str(r['offset']) for r in rows)})\n"
             "each row: [ predicted next-world | GT next-world | goal profile + action pred/GT ]\n"
             "action[0] = (R,U,F metres ; yaw,pitch rad).  action shown = mean over seeds (goal effect).",
             fontsize=10, va="center", family="monospace")

    # per-goal rows
    for i, r in enumerate(rows, start=1):
        axp = fig.add_subplot(gs[i, 0]); axp.imshow(_img(r["pred_w"])); axp.axis("off")
        axp.set_title(f"pred world | goal +{r['offset']}", fontsize=9)
        axg = fig.add_subplot(gs[i, 1]); axg.imshow(_img(r["gt_w"])); axg.axis("off")
        axg.set_title(f"GT world (frame {r['end']})", fontsize=9)

        prof = "  ".join(f"{lbl}={fmt.format(v)}{u}" for (lbl, u, fmt), v in zip(PROFILE_FMT, r["goal_raw"]))
        pa, ga = r["pred_a"][0], r["gt_a"][0]
        a_line = "        " + "  ".join(f"{d:>5}" for d in ADIMS)
        p_line = "pred  " + "  ".join(f"{x:+5.2f}" for x in pa)
        g_line = "gt    " + "  ".join(f"{x:+5.2f}" for x in ga)
        rmse = float(np.sqrt(((r["pred_a"] - r["gt_a"]) ** 2).mean()))
        val = ""
        if r["pred_v"] is not None:
            val = f"\nvalue[0] pred/gt: {float(np.ravel(r['pred_v'])[0]):+.2f} / {float(np.ravel(r['gt_v'])[0]):+.2f}"
        axt = fig.add_subplot(gs[i, 2]); axt.axis("off")
        axt.text(0.0, 0.5,
                 f"GOAL +{r['offset']}  (frame {r['end']})\n{prof}\n\naction[0]:\n{a_line}\n{p_line}\n{g_line}\n"
                 f"chunk action RMSE: {rmse:.3f}{val}",
                 fontsize=8.5, va="center", family="monospace")

    # footer: model's goal signal vs the data's, on pred/gt action[0]
    pa0 = np.stack([r["pred_a"][0] for r in rows]); ga0 = np.stack([r["gt_a"][0] for r in rows])
    model_sig = pa0.std(axis=0).mean(); data_sig = ga0.std(axis=0).mean()
    fig.suptitle(
        f"action[0] spread ACROSS goals  —  model(pred)={model_sig:.3f}   data(GT)={data_sig:.3f}   "
        f"(model/data = {model_sig/(data_sig+1e-9):.2f} of the available goal signal)",
        y=0.005, fontsize=11, weight="bold")
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
