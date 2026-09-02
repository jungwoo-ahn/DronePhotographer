"""Closed-loop DP rollout: observe (Blender render) -> act (DP) -> move -> repeat.

End condition: achieved shot-profile distance to target < goal_thresh (reached),
OR the commanded move stays < conv thresholds for K steps (converged), OR
max_steps. Reports the goal-distance trajectory + start/final frames.

  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/rollout_dp.py \
      --checkpoint runs/<...>/ckpt_best.pt --data_json ... --v6_json ... \
      --target configs/policy/targets/centered_medium.yaml --out_dir /tmp/dp_roll
"""
import argparse, json, subprocess, time, math, os, shutil
from pathlib import Path
import numpy as np, torch, yaml
from src.policy.common.action_repr import ACTION_DIM, POSE_DIM, apply_action_9d, decode_action_9d
from src.policy.common.annotations import iter_windows, iter_goal_start_windows
from src.policy.common.goal_space import goal_keys, normalize_goal, goal_vector
from src.policy.diffusion_policy.dataset import _load_image_as_tensor
from src.policy.diffusion_policy.model import DiffusionPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--config", required=True, help="training config (for model down_dims etc.; "
                "default DiffusionPolicy dims silently mismatch a bigger-UNet checkpoint)")
ap.add_argument("--data_json", required=True)
ap.add_argument("--v6_json", required=True)
ap.add_argument("--target", default=None, help="target-profile YAML (omit with --own_goal)")
ap.add_argument("--own_goal", action="store_true",
                help="sanity check: target = w0's OWN achieved goal (reachable from w0.start by "
                     "construction). Isolates closed-loop navigation from unreachable/OOD targets.")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--blender", default="/home/nas5/jungwooahn/projects/DronePhotographer/blender/blender")  # 4.5.2 LTS (scenes are saved with it; 4.4.3 can't read them)
ap.add_argument("--repo_id", default="facebook/dinov2-large")
ap.add_argument("--max_steps", type=int, default=48)
ap.add_argument("--n_steps", type=int, default=16)
ap.add_argument("--chunk_size", type=int, default=8)
ap.add_argument("--ignore_shoot", type=int, default=0, help="1 = ignore the shoot channel for termination (diagnose pose navigation alone)")
ap.add_argument("--goal_thresh", type=float, default=0.12)
ap.add_argument("--conv_trans", type=float, default=0.03)
ap.add_argument("--conv_rot_deg", type=float, default=1.0)
ap.add_argument("--conv_k", type=int, default=3)
ap.add_argument("--resolution", nargs=2, type=int, default=[480, 720])
a = ap.parse_args()
dev = torch.device("cuda"); dt = torch.bfloat16
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ctl = out / "ctl"; ctl.mkdir(exist_ok=True)
for f in ctl.glob("*"): f.unlink()

if a.own_goal:
    # target = w0's own achieved goal (set after w0 below); keys = the model's goal keys.
    tkeys = goal_keys(yaml.safe_load(Path(a.config).read_text()).get("data", {}).get("goal_score_keys"))
    tnorm = None
else:
    tgt = yaml.safe_load(Path(a.target).read_text())["target"]
    remap = {"center_x": "object_center_x", "center_y": "object_center_y"}
    tgt = {remap.get(k, k): float(v) for k, v in tgt.items()}
    tkeys = goal_keys(list(tgt.keys()))
    tnorm = normalize_goal(np.array([tgt[k] for k in tkeys], np.float32), tkeys)

mcfg = yaml.safe_load(Path(a.config).read_text()).get("model", {})   # must match training UNet
from transformers import AutoImageProcessor, AutoModel
bb = AutoModel.from_pretrained(a.repo_id, torch_dtype=dt); pr = AutoImageProcessor.from_pretrained(a.repo_id)
policy = DiffusionPolicy(
    bb, goal_dim=len(tkeys), chunk_size=a.chunk_size, processor=pr,
    goal_embed_dim=mcfg.get("goal_embed_dim", 128),
    down_dims=tuple(mcfg.get("down_dims", [128, 256, 512])),
    diffusion_step_embed_dim=mcfg.get("diffusion_step_embed_dim", 128),
    num_train_timesteps=mcfg.get("num_train_timesteps", 100),
    beta_schedule=mcfg.get("beta_schedule", "squaredcos_cap_v2"),
).to(dev).eval()
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
missing, unexpected = policy.load_state_dict(ck["policy_state"], strict=False)
_crit = [k for k in (list(missing) + list(unexpected)) if "backbone" not in k]
if _crit:
    raise SystemExit(f"checkpoint/model MISMATCH (non-backbone): {_crit[:6]} ... — config down_dims wrong?")
print(f"DP loaded iter {ck.get('iteration')} (down_dims={tuple(mcfg.get('down_dims',[128,256,512]))})", flush=True)


def goal_distance(achieved):
    av = np.array([achieved[k] for k in tkeys], np.float32)
    return float(np.mean(np.abs(normalize_goal(av, tkeys) - tnorm)))


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


if a.own_goal:
    w0 = next(iter(iter_goal_start_windows(a.data_json, chunk_size=a.chunk_size)))
    _gv = goal_vector(w0.goal_frame.raw, tkeys)
    tnorm = normalize_goal(_gv, tkeys)                     # w0's own reachable goal
    tgt = {k: round(float(v), 1) for k, v in zip(tkeys, _gv)}
    print("OWN-GOAL target:", tgt, flush=True)
else:
    w0 = next(iter(iter_windows(a.data_json, chunk_size=a.chunk_size, stride=1)))
pose = {"pos": list(map(float, w0.start.camera_position)), "forward": list(map(float, w0.start.camera_forward)),
        "up": list(map(float, w0.start.camera_up))}
traj = []; conv = 0; reason = "max_steps"
for t in range(a.max_steps):
    obs = render_score(t, pose)
    gd = goal_distance(obs["achieved"])
    img = _load_image_as_tensor(Path(obs["image"]), tuple(a.resolution)).unsqueeze(0)
    batch = {"state_image": img, "goal_vec": torch.from_numpy(tnorm).unsqueeze(0),
             "action_chunk": torch.zeros(1, a.chunk_size, ACTION_DIM)}
    with torch.no_grad(), torch.amp.autocast(dev.type, dtype=dt):
        oi, g, _ = policy.prepare_inputs(batch, dev, dt)
        a0 = policy.sample(oi, g, n_steps=a.n_steps).pred_action_chunk.squeeze(0).float().cpu().numpy()[0]
    # a0 = [Δtranslation(3), rot6d(6), shoot]; shoot>0.5 = policy declares arrival.
    _dt, _dR = decode_action_9d(a0)
    mt = float(np.linalg.norm(a0[:3]))
    mr = float(math.degrees(np.arccos(np.clip((np.trace(_dR) - 1.0) / 2.0, -1.0, 1.0))))
    shoot = float(a0[POSE_DIM]) if a0.shape[0] > POSE_DIM else 0.0
    traj.append({"t": t, "goal_dist": gd, "occ": obs["occ"], "in_frame": obs["in_frame"],
                 "move_cm": mt * 100, "rot_deg": mr, "shoot": shoot, "image": obs["image"]})
    print(f"t={t:02d} goal_dist={gd:.3f} occ={obs['occ']:.2f} in_frame={obs['in_frame']} move={mt*100:.1f}cm shoot={shoot:.2f}", flush=True)
    if shoot > 0.5 and not a.ignore_shoot: reason = "shoot_declared"; break
    if gd < a.goal_thresh: reason = "goal_reached"; break
    conv = conv + 1 if (mt < a.conv_trans and mr < a.conv_rot_deg) else 0
    if conv >= a.conv_k: reason = "converged"; break
    if t == a.max_steps - 1: break
    p, f, u = apply_action_9d(np.array(pose["pos"], np.float32), np.array(pose["forward"], np.float32),
                              np.array(pose["up"], np.float32), a0)
    pose = {"pos": p.tolist(), "forward": f.tolist(), "up": u.tolist()}

(ctl / "stop.flag").write_text("ok"); time.sleep(2); srv.terminate()
shutil.copy(traj[0]["image"], out / "initial.jpg"); shutil.copy(traj[-1]["image"], out / "final.jpg")
json.dump({"reason": reason, "n_steps": len(traj), "target": tgt,
           "goal_dist_start": traj[0]["goal_dist"], "goal_dist_end": traj[-1]["goal_dist"],
           "goal_dist_min": min(s["goal_dist"] for s in traj), "trajectory": traj},
          open(out / "rollout.json", "w"), indent=2)
print(f"\nEND: reason={reason} steps={len(traj)} goal_dist {traj[0]['goal_dist']:.3f} -> "
      f"{traj[-1]['goal_dist']:.3f} (min {min(s['goal_dist'] for s in traj):.3f})", flush=True)
