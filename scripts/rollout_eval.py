"""Closed-loop Blender evaluation of a goal-conditioned Cosmos policy.

For each held-out placement x goal: drop the policy into the scene, then run a
receding-horizon loop — render -> encode -> policy.sample(goal) -> apply the first
predicted action -> re-render — measuring at every step the geometric achieved shot
profile (src.scoring.projection.score_pose, validated against the dataset) vs the
goal. No MPC / planning: each step is one policy.sample forward pass.

Outputs per-rollout trajectories (frame strip + per-step JSON) and an aggregate
summary.json (per-goal & overall success@tolerance, mean final distance, mean
improvement over the no-op baseline).

  # plumbing smoke (no Blender):
  PYTHONPATH=. .venv/bin/python scripts/rollout_eval.py --mock \
      --checkpoint runs/<ts>_cosmos_2b/ckpt_last.pt --num-placements 1 --max-steps 3
  # real eval (worker, via sbatch): see scripts/sbatch_rollout_eval.sh
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from src.policy.common.action_repr import ACTION_DIM
from src.policy.common.blender_env import (
    BlenderRolloutEnv,
    MockRenderer,
    PersistentBlenderRenderer,
    SubprocessBlenderRenderer,
)
from src.policy.common.goal_space import goal_keys, normalize_goal
from src.policy.common.reward import CameraIntrinsics, geometric_profile_distance, score_distance
from src.policy.common.validation_sample import load_validation_sample
from src.policy.cosmos.model import CosmosWorldActionPolicy
from src.policy.cosmos.vae import CosmosVAEWrapper
from src.scoring.bbox_control import V5_SCORE_KEYS
from src.scoring.projection import project_verts_to_bbox, score_pose

REPO = Path(__file__).resolve().parents[1]

# Per-key success tolerances (raw V5 units). A rollout "reaches" the goal when
# every key is within tolerance.
SUCCESS_TOL = {
    "occupancy": 10, "body_in_frame_ratio": 15,
    "cam_to_obj_azimuth_deg": 20, "cam_to_obj_elevation_deg": 15,
    "object_center_x": 100, "object_center_y": 80,
    "bbox_x_offset": 60, "bbox_y_offset": 60,
}


# --------------------------------------------------------------------------- #
# model loading (faithful to training)
# --------------------------------------------------------------------------- #
def _resolved_config(ckpt_path: Path) -> dict:
    cfg_path = ckpt_path.parent / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    print(f"[warn] {cfg_path} missing — falling back to cosmos_2b.yaml defaults")
    return yaml.safe_load((REPO / "configs/policy/cosmos_2b.yaml").read_text())


def load_policy(ckpt_path: Path, device, dtype):
    """Rebuild the policy with the EXACT training architecture, then load weights.

    eval_cosmos_policy.py's default construction + strict=False silently drops
    mismatched conditioner/action keys — we replicate train_cosmos_policy.py.
    """
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel
    from torch import nn

    from src.policy.cosmos.edm import FlowConfig

    cfg = _resolved_config(ckpt_path)
    bk, data = cfg["backbone"], cfg["data"]
    repo_id, revision = bk["repo_id"], bk.get("revision", "diffusers/base/post-trained")

    transformer = CosmosTransformer3DModel.from_pretrained(
        repo_id, subfolder="transformer", revision=revision, torch_dtype=dtype)
    raw_vae = AutoencoderKLWan.from_pretrained(
        repo_id, subfolder="vae", revision=revision, torch_dtype=dtype)
    vae = CosmosVAEWrapper(raw_vae).to(device).eval()

    crossattn_dim = 1024
    proj = getattr(transformer, "crossattn_proj", None)
    if proj is not None:
        crossattn_dim = next(m.in_features for m in proj.modules() if isinstance(m, nn.Linear))

    flow_dict = cfg.get("flow", {})
    flow_cfg = FlowConfig(**{k: v for k, v in flow_dict.items() if k in FlowConfig.__dataclass_fields__})
    loss = cfg.get("loss", {})
    keys = goal_keys(data["goal_score_keys"])
    policy = CosmosWorldActionPolicy(
        transformer,
        crossattn_dim=crossattn_dim,
        goal_dim=len(keys),
        n_goal_tokens=bk["n_goal_tokens"],
        freeze_backbone=bk["freeze_backbone"],
        anchor_path=cfg.get("conditioner", {}).get("anchor_path"),
        chunk_size=data["chunk_size"],
        lambda_world=float(loss.get("lambda_world", 1.0)),
        lambda_action=float(loss.get("lambda_action", 1.0)),
        lambda_value=float(loss.get("lambda_value", 1.0)),
        flow_config=flow_cfg,
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ckpt["policy_state"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint/architecture mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]} — the policy was built with the wrong "
            "shape; check config.yaml / chunk_size / n_goal_tokens.")
    return policy, vae, keys, int(data["chunk_size"]), int(ckpt.get("iteration", -1))


# --------------------------------------------------------------------------- #
# subject geometry (Blender once per placement, or synthetic for --mock)
# --------------------------------------------------------------------------- #
def extract_geom(run_info_path: str, blender_bin: str) -> dict:
    binp = (REPO / blender_bin).resolve()
    if not binp.exists():
        raise FileNotFoundError(f"Blender binary not found at {binp}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "geom.json"
        cmd = [str(binp), "-b", "-P", str(REPO / "scripts/extract_subject_geom.py"), "--",
               "--run_info_path", str(run_info_path), "--output_json", str(out)]
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"geom extraction failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
        g = json.loads(out.read_text())
    g["verts_world"] = np.asarray(g["verts_world"], dtype=np.float64)
    return g


def mock_geom(subject_center) -> dict:
    c = np.asarray(subject_center, dtype=np.float64)
    offs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=np.float64)
    # frame bounds at z=1 for the v7 24mm / 12.8x9.6mm lens (analytic, AUTO fit ~ width)
    return {"verts_world": c + offs, "frame_bounds": [-0.26667, 0.26667, -0.2, 0.2],
            "render_width": 1024, "render_height": 768}


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def preprocess(pil_image, device, dtype) -> torch.Tensor:
    """PIL RGB -> (1, 3, 480, 720) in [-1, 1], mirroring dataset._load_image_as_tensor."""
    img = pil_image.convert("RGB").resize((720, 480))  # PIL size is (W, H)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device, dtype=dtype)


def achieved_profile(pose, geom, subject_center) -> dict:
    return score_pose(pose["position"], pose["forward"], pose["up"],
                      geom["verts_world"], subject_center,
                      geom["render_width"], geom["render_height"], geom["frame_bounds"])


def dist_raw(achieved: dict, goal: dict, keys) -> float:
    a = np.array([achieved[k] for k in keys], dtype=np.float32)
    g = np.array([goal[k] for k in keys], dtype=np.float32)
    return score_distance(a, g, keys)  # normalizes per-key internally


def is_success(achieved: dict, goal: dict) -> bool:
    return all(abs(achieved[k] - goal[k]) <= SUCCESS_TOL[k] for k in SUCCESS_TOL)


@torch.no_grad()
def rollout(env, policy, vae, geom, goal, keys, subject_center, *, max_steps, execute_k,
            n_steps, device, dtype, intr, start_pose=None):
    goal_norm = torch.from_numpy(
        normalize_goal(np.array([goal[k] for k in keys], dtype=np.float32), keys)
    ).unsqueeze(0).to(device, dtype=dtype)

    obs = (env.reset(start_pose[0], start_pose[1], start_pose[2], render=True)
           if start_pose is not None else env.reset_to_start(0, render=True))
    a0 = achieved_profile(obs["pose"], geom, subject_center)
    d0 = dist_raw(a0, goal, keys)                       # no-op baseline
    steps = [{"t": 0, "profile": a0, "bbox": _bbox(obs["pose"], geom), "distance": d0,
              "geo": geometric_profile_distance(a0, goal, intr)}]
    frames = [obs["image"]]
    reached = is_success(a0, goal)

    t = 0
    while t < max_steps and not reached:
        state = preprocess(obs["image"], device, dtype)
        # Match training: the transformer is bf16 but the conditioner is fp32, so
        # autocast reconciles the mixed dtypes (disabled for the cpu/fp32 mock path).
        with torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            latent = vae.encode_pair_frames(state, state)   # (1,16,2,h,w) — matches training
            out = policy.sample(image_latent=latent, goal_vec=goal_norm, n_steps=n_steps)
        chunk = out.pred_action_chunk.squeeze(0).float().cpu().numpy()   # (chunk, 5)
        for a in chunk[:max(1, execute_k)]:
            obs, _ = env.step(a, render=True)
            t += 1
        ach = achieved_profile(obs["pose"], geom, subject_center)
        steps.append({"t": t, "profile": ach, "bbox": _bbox(obs["pose"], geom),
                      "distance": dist_raw(ach, goal, keys),
                      "geo": geometric_profile_distance(ach, goal, intr)})
        frames.append(obs["image"])
        reached = is_success(ach, goal)

    final = steps[-1]
    return {
        "n_steps": t, "reached": bool(reached),
        "d_start": d0, "d_final": final["distance"], "improvement": d0 - final["distance"],
        "geo_start": steps[0]["geo"], "geo_final": final["geo"],
        "steps": steps,
    }, frames


def save_strip(frames, path: Path) -> None:
    from PIL import Image
    if not frames or any(f is None for f in frames):
        return
    thumbs = [f.convert("RGB").resize((192, 128)) for f in frames]
    strip = Image.new("RGB", (192 * len(thumbs), 128), (0, 0, 0))
    for i, th in enumerate(thumbs):
        strip.paste(th, (i * 192, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)


def _bbox(pose, geom):
    return project_verts_to_bbox(
        geom["verts_world"], pose["position"], pose["forward"], pose["up"],
        geom["render_width"], geom["render_height"], geom["frame_bounds"])


def save_gif(frames, steps, goal_name, path: Path, *, render_w, render_h, fps=2) -> None:
    """Animated drone view per rollout: achieved subject bbox (green) + step/distance overlay."""
    from PIL import Image, ImageDraw
    if not frames or any(f is None for f in frames):
        return
    disp_w = 480
    scale = disp_w / render_w
    imgs = []
    for i, f in enumerate(frames):
        im = f.convert("RGB").resize((disp_w, max(1, int(render_h * scale))))
        dr = ImageDraw.Draw(im)
        st = steps[min(i, len(steps) - 1)]
        bb = st.get("bbox")
        if bb:
            dr.rectangle([bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale],
                         outline=(0, 230, 0), width=3)
        dr.rectangle([0, 0, disp_w, 30], fill=(0, 0, 0))
        dr.text((6, 3), f"step {st['t']}   dist {st['distance']:.3f}", fill=(255, 220, 0))
        dr.text((6, 16), goal_name[:54], fill=(150, 210, 255))
        imgs.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [int(1000 / fps)] * len(imgs)
    durations[-1] = 1600  # hold the final frame
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)


def held_out_placements(data_root: Path, val_names_path: str) -> list[Path]:
    val = {ln.strip() for ln in Path(val_names_path).read_text().splitlines()
           if ln.strip() and not ln.strip().startswith("#")}
    out = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and d.name.split("__")[0] in val and (d / "data.json").exists():
            out.append(d)
    return out


def sample_placements(data_root: Path, val_names_path: str, scenes_mode: str,
                      n: int, rng) -> list[Path]:
    """Pick n placements, round-robin across DISTINCT scenes for diversity.

    scenes_mode='val' restricts to held-out val scenes (generalization metric);
    'all' draws from every scene (a behaviour probe over diverse environments).
    """
    from collections import defaultdict
    pls = [d for d in sorted(data_root.iterdir()) if d.is_dir() and (d / "data.json").exists()]
    if scenes_mode == "val":
        val = {ln.strip() for ln in Path(val_names_path).read_text().splitlines()
               if ln.strip() and not ln.strip().startswith("#")}
        pls = [d for d in pls if d.name.split("__")[0] in val]
    by_scene: dict[str, list[Path]] = defaultdict(list)
    for d in pls:
        by_scene[d.name.split("__")[0]].append(d)
    scenes = list(by_scene)
    rng.shuffle(scenes)
    queues = {sc: [by_scene[sc][i] for i in rng.permutation(len(by_scene[sc]))] for sc in scenes}
    picks: list[Path] = []
    while len(picks) < n and any(queues.values()):
        for sc in scenes:
            if queues[sc]:
                picks.append(queues[sc].pop())
                if len(picks) >= n:
                    break
    return picks


def far_start_pose(subject_center, recorded_start_pos, radius: float):
    """Camera `radius` m from the subject, aimed at it, along the recorded start's
    (occlusion-free) direction — a genuinely-far start so the policy must dolly in."""
    c = np.asarray(subject_center, dtype=np.float64)
    d = np.asarray(recorded_start_pos, dtype=np.float64) - c
    d /= (np.linalg.norm(d) + 1e-9)
    pos = c + radius * d
    fwd = -d                                            # aim back at the subject
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:                    # looking near-vertical
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return pos.astype(np.float32), fwd.astype(np.float32), up.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--goals", default="configs/eval/eval_goals.json", type=Path)
    ap.add_argument("--val-names", default="configs/policy/val_scenes.txt")
    ap.add_argument("--data-root", default="data/trajectories", type=Path)
    ap.add_argument("--vlm-v6-dir", default="data/vlm_object_placing_v6_260428_061326")
    ap.add_argument("--num-placements", type=int, default=4)
    ap.add_argument("--scenes", choices=["all", "val"], default="all",
                    help="all = placements across every scene (diverse probe); val = held-out scenes only")
    ap.add_argument("--start-radius", type=float, default=12.0,
                    help="start the camera this many m from the subject, aimed at it (far start so it must approach); 0 = recorded start pose")
    ap.add_argument("--max-goals", type=int, default=None, help="cap number of goals (quick tests)")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--execute-k", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=32, help="diffusion sampler steps")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--blender-bin", default="blender/blender")
    ap.add_argument("--render-samples", type=int, default=16, help="Cycles samples for eval renders (dataset=32; lower=faster)")
    ap.add_argument("--render-device", default="GPU", help="Blender Cycles device: GPU|CPU")
    ap.add_argument("--render-timeout", type=float, default=900.0, help="per-frame Blender timeout (first GPU render compiles kernels for sm_100)")
    ap.add_argument("--renderer", choices=["persistent", "subprocess"], default="persistent",
                    help="persistent = one Blender per scene (fast); subprocess = one per frame")
    ap.add_argument("--mock", action="store_true", help="MockRenderer + synthetic geom (plumbing, no Blender)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir) if args.out_dir else args.checkpoint.parent / "rollout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, vae, keys, chunk_size, iteration = load_policy(args.checkpoint, device, dtype)
    print(f"loaded policy (iter {iteration}, chunk_size={chunk_size}) | goal keys: {keys}")

    goals = json.loads(Path(args.goals).read_text())
    if args.max_goals:
        goals = goals[:args.max_goals]
    rng = np.random.RandomState(args.seed)
    placements = sample_placements(args.data_root, args.val_names, args.scenes, args.num_placements, rng)
    n_scenes = len({p.name.split("__")[0] for p in placements})
    start_desc = f"far r={args.start_radius:.0f}m" if args.start_radius > 0 else "recorded"
    print(f"placements: {len(placements)} across {n_scenes} distinct scenes ({args.scenes}) | "
          f"start: {start_desc} | goals: {len(goals)}")

    if args.mock:
        renderer = MockRenderer()
    elif args.renderer == "subprocess":
        renderer = SubprocessBlenderRenderer(blender_bin=args.blender_bin, timeout_s=args.render_timeout)
    else:
        renderer = PersistentBlenderRenderer(blender_bin=args.blender_bin, render_timeout_s=args.render_timeout)
    results = []
    for pl in placements:
        sample = load_validation_sample(pl / "data.json", args.vlm_v6_dir, require_vlm=not args.mock)
        sample.render_samples = args.render_samples       # eval speed: GPU + fewer samples
        sample.render_device = args.render_device
        env = BlenderRolloutEnv.from_validation_sample(sample, renderer)
        intr = CameraIntrinsics.from_render(sample.render_width, sample.render_height,
                                            sample.focal_length, sample.sensor_width, sample.sensor_height)
        try:
            geom = mock_geom(sample.subject_center) if args.mock else extract_geom(env.run_info_path, args.blender_bin)
            start_pose = None
            if args.start_radius > 0:
                rec_pos, _, _ = sample.start_pose(0)
                start_pose = far_start_pose(sample.subject_center, rec_pos, args.start_radius)
            for goal in goals:
                summ, frames = rollout(
                    env, policy, vae, geom, goal["profile"], keys, sample.subject_center,
                    max_steps=args.max_steps, execute_k=args.execute_k, n_steps=args.n_steps,
                    device=device, dtype=dtype, intr=intr, start_pose=start_pose)
                tag = f"{pl.name[:40]}__{goal['name']}"
                (out_dir / f"{tag}.json").write_text(json.dumps(
                    {"placement": pl.name, "goal": goal["name"], "goal_profile": goal["profile"], **summ}, indent=2))
                if not args.mock:
                    save_strip(frames, out_dir / "strips" / f"{tag}.png")
                    save_gif(frames, summ["steps"], goal["name"], out_dir / "gifs" / f"{tag}.gif",
                             render_w=sample.render_width, render_h=sample.render_height)
                results.append({"placement": pl.name, "goal": goal["name"],
                                "reached": summ["reached"], "d_start": summ["d_start"],
                                "d_final": summ["d_final"], "improvement": summ["improvement"],
                                "geo_final": summ["geo_final"], "n_steps": summ["n_steps"]})
                print(f"  {tag:64s} reached={summ['reached']!s:5s} "
                      f"d {summ['d_start']:.3f}->{summ['d_final']:.3f} (improve {summ['improvement']:+.3f})")
        finally:
            env.close()

    n = len(results)
    summary = {
        "checkpoint": str(args.checkpoint), "iteration": iteration, "n_rollouts": n,
        "success_rate": float(np.mean([r["reached"] for r in results])) if n else 0.0,
        "mean_d_final": float(np.mean([r["d_final"] for r in results])) if n else None,
        "mean_improvement_over_noop": float(np.mean([r["improvement"] for r in results])) if n else None,
        "mean_geo_final": float(np.mean([r["geo_final"] for r in results])) if n else None,
        "per_goal": {},
        "results": results,
        "config": {"max_steps": args.max_steps, "execute_k": args.execute_k,
                   "n_steps": args.n_steps, "success_tol": SUCCESS_TOL, "mock": args.mock},
    }
    for g in sorted({r["goal"] for r in results}):
        sub = [r for r in results if r["goal"] == g]
        summary["per_goal"][g] = {
            "success_rate": float(np.mean([r["reached"] for r in sub])),
            "mean_improvement": float(np.mean([r["improvement"] for r in sub])),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir/'summary.json'}")
    print(f"  rollouts={n}  success@tol={summary['success_rate']:.2f}  "
          f"mean d_final={summary['mean_d_final']}  mean improve={summary['mean_improvement_over_noop']}")


if __name__ == "__main__":
    sys.exit(main())
