"""Closed-loop eval of OUR baselines on jungwoo's V12 metric, so the numbers are directly
comparable to Cosmos3.

Ports the protocol of `scripts/closed_loop_eval.py` from jungwoo-ahn/DronePhotographerV12.
The distance itself needs no porting: `src/policy/common/reward.py` already contains
`pose_to_geometry` / `_geometry_distance` byte-identical to V12's `src/common/reward.py`.

    d = sqrt( great_circle(az,el)^2 + (size_a - size_g)^2 + hypot(daim_x, daim_y)^2 )   [radians]

It is pure pose math, so Blender rendering is only needed to give the policy an observation.

Protocol (V12 closed_loop_eval.py):
  - horizon: ceil(|goal_frame_idx - start_frame_idx| / chunk_size) chunks, + --extra-chunks
  - per chunk: render -> predict 10D chunk -> if max(chunk[:,9]) > shoot_threshold, STOP
    BEFORE executing -> else apply all chunk_size pose steps (no render between) ->
    re-render -> record d
  - headline: improvement = d_start - d_end (a no-op scores exactly 0 by construction)
  - summary: mean/median improvement, frac_positive, mean_best_improvement,
    frac_best_positive, mean_d_start, mean_d_end -- plus --split {val,train} as the control.

  PYTHONPATH=. python scripts/closed_loop_eval_baselines.py --policy pi05 \
      --checkpoint runs/pi05_fair/checkpoints/190000/pretrained_model --split val --episodes 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.policy.common.action_repr import POSE_DIM, apply_action_9d
from src.policy.common.annotations import iter_goal_start_windows
from src.policy.common.goal_space import goal_keys, goal_vector, normalize_goal
from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt
from src.policy.common.reward import _geometry_distance, pose_to_geometry

REPO = Path(__file__).resolve().parents[1]
V6_DIR = REPO / "data/vlm_object_placing_v6_260428_061326"


def geometry_distance(position, forward, up, goal_view) -> float:
    """V12 `closed_loop_eval.geometry_distance`: achieved-vs-goal geometry, in radians."""
    achieved = pose_to_geometry(position, forward, up,
                               subject_center=goal_view.subject_center,
                               subject_height=goal_view.subject_height)
    goal = pose_to_geometry(goal_view.camera_position, goal_view.camera_forward,
                            goal_view.camera_up,
                            subject_center=goal_view.subject_center,
                            subject_height=goal_view.subject_height)
    return float(_geometry_distance(achieved, goal))


# --------------------------------------------------------------------------- policies
class DPPolicy:
    """Diffusion Policy: goal arrives as a NORMALIZED VECTOR."""

    def __init__(self, ckpt, cfg_path, device, n_steps=16, chunk=8, resolution=(480, 720)):
        import yaml
        from transformers import AutoImageProcessor, AutoModel
        from src.policy.diffusion_policy.model import DiffusionPolicy
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.keys = goal_keys(cfg.get("data", {}).get("goal_score_keys"))
        m = cfg.get("model", {})
        self.dt, self.dev, self.n_steps, self.chunk, self.res = torch.bfloat16, device, n_steps, chunk, resolution
        bb = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=self.dt)
        pr = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.policy = DiffusionPolicy(
            bb, goal_dim=len(self.keys), chunk_size=chunk, processor=pr,
            goal_embed_dim=m.get("goal_embed_dim", 128),
            down_dims=tuple(m.get("down_dims", [128, 256, 512])),
            diffusion_step_embed_dim=m.get("diffusion_step_embed_dim", 128),
            num_train_timesteps=m.get("num_train_timesteps", 100),
            beta_schedule=m.get("beta_schedule", "squaredcos_cap_v2"),
        ).to(device).eval()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = self.policy.load_state_dict(sd["policy_state"], strict=False)
        bad = [k for k in list(missing) + list(unexpected) if "backbone" not in k]
        if bad:
            raise SystemExit(f"DP checkpoint/model MISMATCH: {bad[:5]} -- wrong --config?")

    def goal(self, w):
        return normalize_goal(goal_vector(w.goal_frame.raw, self.keys), self.keys)

    def act(self, image_path, goal):
        from src.policy.common.action_repr import ACTION_DIM
        from src.policy.diffusion_policy.dataset import _load_image_as_tensor
        img = _load_image_as_tensor(Path(image_path), tuple(self.res)).unsqueeze(0)
        batch = {"state_image": img, "goal_vec": torch.from_numpy(goal).unsqueeze(0),
                 "action_chunk": torch.zeros(1, self.chunk, ACTION_DIM)}
        with torch.no_grad(), torch.amp.autocast(self.dev.type, dtype=self.dt):
            oi, g, _ = self.policy.prepare_inputs(batch, self.dev, self.dt)
            return self.policy.sample(oi, g, n_steps=self.n_steps).pred_action_chunk.squeeze(0).float().cpu().numpy()


class LeRobotPolicy:
    """pi0.5 / GR00T: goal arrives as an English PROMPT (same as Cosmos3)."""

    def __init__(self, kind, ckpt, device, resize=224, state_dim=9):
        from lerobot.policies.factory import make_pre_post_processors
        if kind == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
        else:
            from lerobot.policies.groot.modeling_groot import GrootPolicy as P
        self.policy = P.from_pretrained(ckpt).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, ckpt)
        self.resize, self.state = resize, torch.zeros(state_dim, dtype=torch.float32)

    def goal(self, w):
        return goal_prompt(goal_vector(w.goal_frame.raw, NL_GOAL_KEYS), NL_GOAL_KEYS,
                           crop=w.goal_frame.raw)

    def act(self, image_path, goal):
        from PIL import Image
        with Image.open(image_path) as im:
            a = np.asarray(im.convert("RGB").resize((self.resize, self.resize), Image.BILINEAR),
                           np.float32) / 255.0
        batch = {"observation.images.image": torch.from_numpy(a).permute(2, 0, 1),
                 "observation.state": self.state, "task": goal}
        with torch.no_grad():
            return self.post(self.policy.predict_action_chunk(self.pre(batch))).squeeze(0).float().cpu().numpy()


# --------------------------------------------------------------------------- blender
class Renderer:
    """One persistent Blender server per placement (scripts/rollout_server.py)."""

    def __init__(self, data_json, v6_json, out_dir, blender="blender/blender"):
        self.ctl = Path(out_dir) / "ctl"
        self.ctl.mkdir(parents=True, exist_ok=True)
        for f in self.ctl.glob("*"):
            f.unlink()
        self.proc = subprocess.Popen(
            [blender, "--background", "--python", "scripts/rollout_server.py", "--",
             "--data_json", str(data_json), "--v6_json", str(v6_json),
             "--assets_root", str(REPO), "--ctl_dir", str(self.ctl)],
            stdout=open(Path(out_dir) / "server.log", "w"), stderr=subprocess.STDOUT,
            env={**os.environ})
        for _ in range(900):
            if (self.ctl / "ready.flag").exists():
                return
            if self.proc.poll() is not None:
                raise RuntimeError("blender server died; see server.log")
            time.sleep(1)
        raise RuntimeError("blender server load timeout")

    def render(self, t, pos, fwd, up) -> str:
        req = {"t": t, "pose": {"pos": list(map(float, pos)), "forward": list(map(float, fwd)),
                                "up": list(map(float, up))}}
        (self.ctl / "req.json.tmp").write_text(json.dumps(req))
        (self.ctl / "req.json.tmp").rename(self.ctl / "req.json")
        resp = self.ctl / f"resp_{t:03d}.json"
        for _ in range(12000):
            if resp.exists():
                return json.loads(resp.read_text())["image"]
            time.sleep(0.05)
        raise RuntimeError(f"render timeout t={t}")

    def close(self):
        try:
            (self.ctl / "stop.flag").write_text("ok"); time.sleep(2); self.proc.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------- episodes
def select_placements(split, val_scenes_path, roots, limit):
    vs = frozenset(json.loads(Path(val_scenes_path).read_text())["scenes"])
    fm = json.loads((REPO / "configs/policy/facing_map_final.json").read_text())
    out = []
    for root in roots:
        for d in sorted(os.listdir(root)):
            dj = Path(root) / d / "data.json"
            if not dj.exists() or not (V6_DIR / f"{d}.json").exists():
                continue
            in_val = d.split("__")[0] in vs
            if split != "all" and in_val != (split == "val"):
                continue
            try:                                    # skip objects with no facing map (unscorable)
                obj = Path(json.loads(dj.read_text())["object_file"]).stem
            except Exception:
                continue
            if (fm.get(obj) or {}).get("front_az") is None:
                continue
            out.append((d, dj))
            if len(out) >= limit:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["dp", "pi05", "groot"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="DP training config (required for --policy dp)")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="val = held-out (generalization); train = the control V12 added in be8f72f")
    ap.add_argument("--val-scenes", default="configs/policy/val_scenes.json", dest="val_scenes")
    ap.add_argument("--roots", nargs="+", default=["data/trajectories_full"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=8, dest="chunk_size")
    ap.add_argument("--extra-chunks", type=int, default=0, dest="extra_chunks")
    ap.add_argument("--shoot-threshold", type=float, default=0.5, dest="shoot_threshold")
    ap.add_argument("--stop-on-shoot", type=int, default=1, dest="stop_on_shoot")
    ap.add_argument("--blender", default="blender/blender")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    pol = (DPPolicy(a.checkpoint, a.config, dev, chunk=a.chunk_size) if a.policy == "dp"
           else LeRobotPolicy(a.policy, a.checkpoint, dev))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    places = select_placements(a.split, a.val_scenes, a.roots, a.episodes)
    print(f"{a.policy} | split={a.split} | {len(places)} placements", flush=True)

    results = []
    for i, (name, dj) in enumerate(places):
        try:
            w = next(iter(iter_goal_start_windows(str(dj), chunk_size=a.chunk_size)))
        except StopIteration:
            continue
        delta = abs(w.goal_frame.frame_idx - w.start_frame_idx)
        n_chunks = int(math.ceil(delta / a.chunk_size)) + a.extra_chunks
        g = pol.goal(w)
        pos = np.asarray(w.start.camera_position, np.float32)
        fwd = np.asarray(w.start.camera_forward, np.float32)
        up = np.asarray(w.start.camera_up, np.float32)
        d0 = geometry_distance(pos, fwd, up, w.goal_frame)
        trace, declared = [d0], None
        try:
            rend = Renderer(dj, V6_DIR / f"{name}.json", out / name, a.blender)
        except Exception as e:
            print(f"  [{i}] {name[:40]}: renderer failed ({e})", flush=True); continue
        try:
            img = rend.render(0, pos, fwd, up)
            for c in range(n_chunks):
                chunk = pol.act(img, g)
                shoot = float(np.max(chunk[:, POSE_DIM])) if chunk.shape[1] > POSE_DIM else 0.0
                if declared is None and shoot > a.shoot_threshold:
                    declared = c
                    if a.stop_on_shoot:
                        break                       # stop BEFORE executing (V12 semantics)
                for step in chunk[:, :POSE_DIM]:
                    pos, fwd, up = apply_action_9d(pos, fwd, up, step)
                img = rend.render(c + 1, pos, fwd, up)
                trace.append(geometry_distance(pos, fwd, up, w.goal_frame))
        finally:
            rend.close()
        d1, dbest = trace[-1], float(min(trace))
        results.append({"placement": name, "scene": name.split("__")[0], "n_chunks": n_chunks,
                        "declared_stop": declared, "d_start": d0, "d_end": d1, "d_best": dbest,
                        "improvement": d0 - d1, "best_improvement": d0 - dbest, "trace": trace})
        print(f"  [{i}] {name[:42]:42s} d {d0:.3f}->{d1:.3f} (best {dbest:.3f}) "
              f"improve {d0 - d1:+.3f} stop@{declared}", flush=True)

    imp = np.array([r["improvement"] for r in results], np.float64)
    best = np.array([r["best_improvement"] for r in results], np.float64)
    summary = {
        "policy": a.policy, "checkpoint": a.checkpoint, "split": a.split,
        "n_rollouts": len(results),
        "mean_improvement_over_noop": float(imp.mean()) if len(imp) else None,
        "median_improvement": float(np.median(imp)) if len(imp) else None,
        "frac_positive": float((imp > 0).mean()) if len(imp) else None,
        "mean_best_improvement": float(best.mean()) if len(best) else None,
        "frac_best_positive": float((best > 0).mean()) if len(best) else None,
        "mean_d_start": float(np.mean([r["d_start"] for r in results])) if results else None,
        "mean_d_end": float(np.mean([r["d_end"] for r in results])) if results else None,
        "config": {"chunk_size": a.chunk_size, "extra_chunks": a.extra_chunks,
                   "shoot_threshold": a.shoot_threshold, "stop_on_shoot": bool(a.stop_on_shoot)},
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== {a.policy} [{a.split}] n={len(results)} ===")
    print(f"mean improvement   {summary['mean_improvement_over_noop']}")
    print(f"median improvement {summary['median_improvement']}")
    print(f"frac_positive      {summary['frac_positive']}")
    print(f"mean d_start->d_end {summary['mean_d_start']} -> {summary['mean_d_end']}")
    print(f"SUMMARY_JSON {out / 'summary.json'}")


if __name__ == "__main__":
    main()
