"""Closed-loop sim rollout for the pretrained-VLA baselines (LeRobot pi05 / groot).

Same loop as scripts/rollout_dp.py (observe via Blender -> act -> move -> repeat, score
achieved-vs-target shot profile), but for a LeRobot policy: the goal reaches the model as
the NATURAL-LANGUAGE prompt (goal_text.goal_prompt), not a goal vector. Reuses the fixed
rollout_server.py (renders + scores, emits subject_bearing_deg via the facing map).

  conda activate vla|vla_groot
  CUDA_VISIBLE_DEVICES=N HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    python scripts/rollout_vla.py --policy groot --checkpoint runs/<run>/checkpoints/<N>/pretrained_model \
      --data_json ... --v6_json ... --own_goal --out_dir /tmp/vla_roll --blender blender/blender
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.policy.common.action_repr import POSE_DIM, apply_action_9d, decode_action_9d
from src.policy.common.annotations import iter_goal_start_windows
from src.policy.common.goal_space import goal_keys, goal_vector, normalize_goal
from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, choices=["pi05", "groot"])
ap.add_argument("--checkpoint", required=True, help="a checkpoint's pretrained_model dir")
ap.add_argument("--data_json", required=True)
ap.add_argument("--v6_json", required=True)
ap.add_argument("--target", default=None, help="target-profile YAML (omit with --own_goal)")
ap.add_argument("--own_goal", action="store_true", help="target = w0's own achieved goal (reachable)")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--blender", default="blender/blender")
ap.add_argument("--max_steps", type=int, default=24)
ap.add_argument("--chunk_size", type=int, default=8)
ap.add_argument("--resize", type=int, default=224)
ap.add_argument("--state_dim", type=int, default=9)
ap.add_argument("--ignore_shoot", type=int, default=0)
ap.add_argument("--goal_thresh", type=float, default=0.12)
ap.add_argument("--conv_trans", type=float, default=0.03)
ap.add_argument("--conv_rot_deg", type=float, default=1.0)
ap.add_argument("--conv_k", type=int, default=3)
a = ap.parse_args()

dev = torch.device("cuda")
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ctl = out / "ctl"; ctl.mkdir(exist_ok=True)
for f in ctl.glob("*"): f.unlink()

# --- target profile + its NL prompt --------------------------------------------------
tkeys = goal_keys(NL_GOAL_KEYS)                        # model goal keys (subject_bearing at slot 2)
if a.own_goal:
    w0 = next(iter(iter_goal_start_windows(a.data_json, chunk_size=a.chunk_size)))
    tvec = goal_vector(w0.goal_frame.raw, tkeys)
    prompt = goal_prompt(tvec, tkeys, crop=w0.goal_frame.raw)
else:
    import yaml
    tgt = yaml.safe_load(Path(a.target).read_text())["target"]
    tvec = np.array([float(tgt[k]) for k in tkeys], np.float32)
    prompt = goal_prompt(tvec, tkeys)
    w0 = next(iter(iter_goal_start_windows(a.data_json, chunk_size=a.chunk_size)))
tnorm = normalize_goal(tvec, tkeys)
print(f"target prompt: {prompt}", flush=True)

# --- policy + processors -------------------------------------------------------------
from lerobot.policies.factory import make_pre_post_processors
if a.policy == "pi05":
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Pol
else:
    from lerobot.policies.groot.modeling_groot import GrootPolicy as Pol
policy = Pol.from_pretrained(a.checkpoint).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, a.checkpoint)
print(f"{a.policy} loaded from {a.checkpoint}", flush=True)

VKEY = "observation.images.image"


def _chw(path, s):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((s, s), Image.BILINEAR), np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def goal_distance(achieved):
    av = np.array([achieved[k] for k in tkeys], np.float32)
    return float(np.mean(np.abs(normalize_goal(av, tkeys) - tnorm)))


# --- Blender render+score server (same as DP path) -----------------------------------
srv = subprocess.Popen([a.blender, "--background", "--python", "scripts/rollout_server.py", "--",
    "--data_json", a.data_json, "--v6_json", a.v6_json, "--assets_root", str(Path.cwd()),
    "--ctl_dir", str(ctl)], stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT, env={**os.environ})
for _ in range(600):
    if (ctl / "ready.flag").exists(): break
    if srv.poll() is not None: raise SystemExit("server died; see server.log")
    time.sleep(1)
else:
    raise SystemExit("server load timeout")


def render_score(t, pose):
    (ctl / "req.json.tmp").write_text(json.dumps({"t": t, "pose": pose})); (ctl / "req.json.tmp").rename(ctl / "req.json")
    resp = ctl / f"resp_{t:03d}.json"
    for _ in range(12000):
        if resp.exists(): return json.loads(resp.read_text())
        time.sleep(0.05)
    raise SystemExit(f"render timeout t={t}")


zero_state = torch.zeros(a.state_dim, dtype=torch.float32)
pose = {"pos": list(map(float, w0.start.camera_position)), "forward": list(map(float, w0.start.camera_forward)),
        "up": list(map(float, w0.start.camera_up))}
traj = []; conv = 0; reason = "max_steps"
for t in range(a.max_steps):
    obs = render_score(t, pose)
    gd = goal_distance(obs["achieved"])
    batch = {VKEY: _chw(Path(obs["image"]), a.resize), "observation.state": zero_state, "task": prompt}
    with torch.no_grad():
        chunk = post(policy.predict_action_chunk(pre(batch))).squeeze(0).float().cpu().numpy()
    a0 = chunk[0][:10]
    _dt, _dR = decode_action_9d(a0[:POSE_DIM])
    mt = float(np.linalg.norm(a0[:3]))
    mr = float(math.degrees(np.arccos(np.clip((np.trace(_dR) - 1.0) / 2.0, -1.0, 1.0))))
    shoot = float(a0[POSE_DIM]) if a0.shape[0] > POSE_DIM else 0.0
    traj.append({"t": t, "goal_dist": gd, "occ": obs["occ"], "in_frame": obs["in_frame"],
                 "move_cm": mt * 100, "rot_deg": mr, "shoot": shoot})
    print(f"t={t:02d} goal_dist={gd:.3f} occ={obs['occ']:.2f} in_frame={obs['in_frame']} move={mt*100:.1f}cm shoot={shoot:.2f}", flush=True)
    if shoot > 0.5 and not a.ignore_shoot: reason = "shoot_declared"; break
    if gd < a.goal_thresh: reason = "goal_reached"; break
    conv = conv + 1 if (mt < a.conv_trans and mr < a.conv_rot_deg) else 0
    if conv >= a.conv_k: reason = "converged"; break
    if t == a.max_steps - 1: break
    p, f, u = apply_action_9d(np.array(pose["pos"], np.float32), np.array(pose["forward"], np.float32),
                              np.array(pose["up"], np.float32), a0[:POSE_DIM])
    pose = {"pos": p.tolist(), "forward": f.tolist(), "up": u.tolist()}

(ctl / "stop.flag").write_text("ok"); time.sleep(2); srv.terminate()
g0 = traj[0]["goal_dist"] if traj else float("nan")
gm = min(x["goal_dist"] for x in traj) if traj else float("nan")
gf = traj[-1]["goal_dist"] if traj else float("nan")
json.dump({"reason": reason, "n_steps": len(traj), "target": {k: float(v) for k, v in zip(tkeys, tvec)},
           "goal_dist_start": g0, "goal_dist_final": gf, "goal_dist_min": gm, "traj": traj},
          open(out / "summary.json", "w"), indent=1)
print(f"END: reason={reason} steps={len(traj)} goal_dist {g0:.3f} -> {gf:.3f} (min {gm:.3f})", flush=True)
