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

from types import SimpleNamespace

import numpy as np
import torch

from src.policy.common.action_repr import (POSE_DIM, apply_action_5d, apply_action_9d,
                                           yaw_from_right_positive)
from src.policy.common.annotations import iter_goal_start_windows
from src.policy.common.goal_space import goal_keys, goal_vector, normalize_goal
from src.policy.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH
from src.policy.common.goal_text import NL_GOAL_KEYS, goal_prompt
from src.policy.common.reward import _geometry_distance, pose_to_geometry

REPO = Path(__file__).resolve().parents[1]
GOAL_RENDER_T = 99      # distinct frame index so the goal preview cannot collide with a rollout step
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



class _FiveDPolicy:
    """Base for policies that emit the interpretable 5D action (dx,dy,dz,dyaw,dpitch).

    The learned baselines predict a 10D chunk with rot6d, which neither an LLM nor a 2D
    composition model can express. Both of these instead emit one camera-local 5D move per
    observation, stepped with `apply_action_5d`. The metric is unaffected: `geometry_distance`
    is pure pose math, so a 5D-stepped pose scores identically to a 9D-stepped one.

    Horizon is OBSERVATION-matched, not step-matched: these get `n_chunks` decisions, the same
    number of renders the learned policies see at chunk boundaries. Matching the learned
    policies' 8 pose-steps per chunk instead would give these 8x the decisions (and, for the
    agent, 8x the API calls) for the same visual evidence.
    """
    action_space = "5d"
    # Clamps matched to what the learned policies actually do in ONE chunk (8 pose steps),
    # measured over 867 chunks of data/trajectories_full: translation median 1.62 m / p90 2.44 /
    # p99 4.14, aim change median 16.5 deg / p99 38.1 / max 39.2. llm_policy's inherited 60 deg
    # was 1.5x anything the data ever does.
    max_trans = 3.0
    max_ang_deg = 40.0


class GPT5Agent(_FiveDPolicy):
    """GPT-5 as a closed-loop camera agent, conditioned on TEXT or on a GOAL IMAGE.

    Text mode gets the same natural-language shot brief the VLAs are conditioned on, so it is
    the apples-to-apples row. Image mode gets the goal frame itself, which is strictly more
    information (it reveals the scene, not just the requested composition) and is reported
    separately rather than mixed in.
    """

    wants_profile = False        # set by --feedback: include the measured current profile

    def __init__(self, model, goal_mode="text", feedback=False,
                 cfg_path="configs/policy/llm_policy_api.yaml",
                 max_tokens=8192, temperature=0.2):
        # NB max_tokens is NOT taken from llm_policy_api.yaml: its 2048 was sized for a
        # non-reasoning model (gemini-2.5-flash). gpt-5 spends ~2.1k reasoning tokens on an
        # image + measurement prompt, so at 2048 it returns finish_reason="length" with EMPTY
        # content -- half the decisions in the first feedback smoke were silently zero-actions.
        import yaml
        from openai import OpenAI
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        vc = cfg.get("api") or cfg.get("vlm") or cfg
        key = vc.get("api_key") or os.environ.get(vc.get("api_key_env", "GATEWAY_API_KEY"))
        if not key:
            raise SystemExit("no gateway api_key in config and no GATEWAY_API_KEY in env")
        self.client = OpenAI(base_url=vc["api_base"], api_key=key)
        self.model, self.goal_mode = model, goal_mode
        self.wants_profile = bool(feedback)
        self.max_tokens, self.temperature = max_tokens, temperature
        self.calls = self.failures = 0

    SYSTEM = (
        "You are a camera operator in a 3D scene. You see the CURRENT camera view and a "
        "description of the SHOT you must achieve. Move the camera one step toward that shot.\n"
        "Reply with STRICT JSON only, no prose:\n"
        '{"reasoning":"<one short sentence>","dx":<float>,"dy":<float>,"dz":<float>,'
        '"dyaw_deg":<float>,"dpitch_deg":<float>,"done":<true|false>}\n'
        "Axes are camera-local in metres: dx=right(+)/left(-), dy=up(+)/down(-), "
        "dz=forward(+)/backward(-). dyaw_deg turns right(+)/left(-); dpitch_deg tilts up(+)/down(-). "
        "Limits: |d*| <= 3.0 m, |angles| <= 40 deg.\n"
        "THE LENS IS NARROW: horizontal field of view 29.9 deg, vertical 22.6 deg. The subject is "
        "typically ~5 m away, so a yaw of more than ~15 deg, or a sideways move of more than ~1.3 m "
        "that you do not compensate for, pushes the subject OUT OF FRAME -- and once it is gone you "
        "are searching blind. To change which side of the subject you are shooting from, ORBIT: "
        "strafe with dx AND turn back toward the subject with the opposite-sign dyaw in the SAME "
        "move, so it stays centred. Keep the subject in frame on every single move; prefer a small "
        "correct move over a large one.\n"
        "Set done=true only when the current view already matches the requested shot."
    )

    @property
    def needs_goal_render(self):
        return self.goal_mode == "image"

    def goal(self, ep, goal_image=None):
        """Text mode -> the NL shot brief. Image mode -> the goal pose, rendered live."""
        if self.goal_mode == "image":
            if goal_image is None:
                raise RuntimeError("image goal mode needs a rendered goal frame")
            return {"mode": "image", "image": goal_image, "raw": ep.goal_frame.raw}
        return {"mode": "text", "raw": ep.goal_frame.raw,
                "text": goal_prompt(goal_vector(ep.goal_frame.raw, NL_GOAL_KEYS),
                                    NL_GOAL_KEYS, crop=ep.goal_frame.raw)}

    @staticmethod
    def _b64(path):
        import base64
        return base64.b64encode(Path(path).read_bytes()).decode()

    # Units legend for the measured profile, so the numbers are not bare ints to the agent.
    _FB_KEYS = (
        ("subject_bearing_deg",       "deg around subject, 0-360 (0 = subject's front)"),
        ("cam_to_obj_elevation_deg",  "deg, negative = camera above looking down"),
        ("occupancy",                 "% of frame filled by subject"),
        ("body_in_frame_ratio",       "% of the subject that is inside the frame"),
        ("object_center_x",           "px, subject centre x (frame is 0-%d)" % RENDER_WIDTH),
        ("object_center_y",           "px, subject centre y (frame is 0-%d)" % RENDER_HEIGHT),
    )

    @staticmethod
    def _wrap180(d):
        """Signed shortest angular error; bearing is circular, so 350 -> 10 is +20, not -340."""
        return (float(d) + 180.0) % 360.0 - 180.0

    def _feedback_text(self, achieved, raw):
        cur, tgt, err = {}, {}, {}
        for k, unit in self._FB_KEYS:
            a, g = achieved.get(k), raw.get(k)
            if a is None or g is None:
                continue
            cur[k], tgt[k] = float(a), float(g)
            err[k] = round(self._wrap180(g - a) if k == "subject_bearing_deg"
                           else float(g) - float(a), 1)
        if not err:
            return None
        legend = "; ".join(f"{k} ({u})" for k, u in self._FB_KEYS if k in cur)
        dist = achieved.get("subject_distance_m")
        scale = ""
        if dist:
            # Lets it convert a desired bearing change into a strafe instead of guessing.
            scale = (f"\nSubject is {float(dist):.1f} m away, so a 1 deg change of your angle around "
                     f"it is about {float(dist) * 0.01745:.2f} m of sideways travel, and a sideways "
                     f"move of {float(dist) * 0.2679:.2f} m swings it to the edge of the frame "
                     f"unless you turn back toward it in the same move.")
        return ("MEASUREMENTS (units: " + legend + ")\n"
                "YOUR VIEW NOW: " + json.dumps(cur) + "\n"
                "TARGET: " + json.dumps(tgt) + "\n"
                "REMAINING ERROR (target - now): " + json.dumps(err) + "\n"
                + scale +
                "\nDrive every error toward 0. Note subject_bearing_deg is your angular position "
                "AROUND the subject: it changes only by orbiting (strafe sideways and re-aim), "
                "never by panning in place. occupancy changes by moving along dz.")

    def act(self, image_path, goal, achieved=None):
        content = [{"type": "text", "text": "CURRENT VIEW:"},
                   {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._b64(image_path)}"}}]
        if goal["mode"] == "image":
            content += [{"type": "text", "text": "TARGET SHOT (reproduce this viewpoint):"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{self._b64(goal['image'])}"}}]
        else:
            content.append({"type": "text", "text": f"TARGET SHOT: {goal['text']}"})
        if self.wants_profile and achieved and goal.get("raw"):
            # Without this the agent gets a quantitative brief but no measurement of its OWN
            # bearing/occupancy, so it cannot tell which direction reduces the error and searches
            # blindly. This is the Reviewer signal a real agentic system (PhotoFlow) would have.
            fb = self._feedback_text(achieved, goal["raw"])
            if fb:
                content.append({"type": "text", "text": fb})
        self.calls += 1
        msgs = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": content}]
        txt, why = "", ""
        for budget in (self.max_tokens, self.max_tokens * 2):   # retry once if reasoning ran long
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=budget, temperature=self.temperature,
                    messages=msgs)
            except Exception as e:
                self.failures += 1
                return np.zeros(5, np.float32), 0.0, {"error": str(e)[:160]}
            txt = r.choices[0].message.content or ""
            why = r.choices[0].finish_reason
            if txt.strip():
                break
        if not txt.strip():
            self.failures += 1
            return np.zeros(5, np.float32), 0.0, {"empty": why}
        d = _parse_json_obj(txt)
        if d is None:
            self.failures += 1
            return np.zeros(5, np.float32), 0.0, {"unparsed": txt[:160], "finish": why}
        a = np.zeros(5, np.float32)
        try:
            for i, k in enumerate(("dx", "dy", "dz")):
                a[i] = float(np.clip(float(d.get(k, 0.0)), -self.max_trans, self.max_trans))
            ang = np.deg2rad(self.max_ang_deg)
            # The prompt speaks human ("dyaw_deg turns right(+)"), the repr's dyaw is left-positive.
            a[3] = float(np.clip(yaw_from_right_positive(np.deg2rad(float(d.get("dyaw_deg", 0.0)))),
                                 -ang, ang))
            a[4] = float(np.clip(np.deg2rad(float(d.get("dpitch_deg", 0.0))), -ang, ang))
        except (TypeError, ValueError):
            self.failures += 1
            return np.zeros(5, np.float32), 0.0, {"bad_fields": str(d)[:160]}
        return a, (1.0 if d.get("done") is True else 0.0), {"reasoning": str(d.get("reasoning", ""))[:120]}


class GatewayVLM:
    """Minimal `ask_json` over the LetSur OpenAI-compatible gateway, for PhotoFlow's roles."""

    def __init__(self, model, cfg_path="configs/policy/llm_policy_api.yaml",
                 max_tokens=8192, temperature=0.2):
        import yaml
        from openai import OpenAI
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        vc = cfg.get("api") or cfg.get("vlm") or cfg
        key = vc.get("api_key") or os.environ.get(vc.get("api_key_env", "GATEWAY_API_KEY"))
        if not key:
            raise SystemExit("no gateway api_key in config and no GATEWAY_API_KEY in env")
        self.client = OpenAI(base_url=vc["api_base"], api_key=key)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    @staticmethod
    def _img(path):
        import base64
        return {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(Path(path).read_bytes()).decode()}"}}

    def ask_json(self, system, user, image=None, images=()):
        content = [{"type": "text", "text": user}]
        for item in images or ():                      # labelled establishing shots
            path, label = item if isinstance(item, (tuple, list)) else (item, None)
            if label:
                content.append({"type": "text", "text": label})
            content.append(self._img(path))
        if image:
            content.append(self._img(image))
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        for budget in (self.max_tokens, self.max_tokens * 2):   # reasoning models run long
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=budget, temperature=self.temperature,
                    messages=msgs)
            except Exception:
                return None
            txt = r.choices[0].message.content or ""
            if txt.strip():
                return _parse_json_obj(txt)
        return None


class PhotoFlowPolicy:
    """PhotoFlow-style agentic search, adapted to this harness (see src/policy/photoflow/).

    Unlike every other row this is a GLOBAL SEARCH, not an incremental policy: it proposes
    absolute camera states and keeps the best, so `improvement = d_start - d_end` stops
    measuring navigation and starts measuring "can a search find the requested viewpoint".
    Its render count is reported in the summary because that, not the algorithm, is the main
    confound against policies that get `n_chunks` renders.
    """

    search_space = "global"

    def __init__(self, model, render_budget=None, rounds=None, candidates=None):
        from src.policy.photoflow.agent import (CANDIDATES_PER_ROUND, ROUNDS, PhotoFlowAgent,
                                                pose_from_look_at)
        self._pose = pose_from_look_at
        self.agent = PhotoFlowAgent(GatewayVLM(model), rounds=rounds or ROUNDS,
                                    candidates=candidates or CANDIDATES_PER_ROUND,
                                    render_budget=render_budget)

    def goal(self, ep):
        """The brief, plus the composition point/half the brief's placement words imply."""
        raw = ep.goal_frame.raw
        cx = float(raw.get("object_center_x", RENDER_WIDTH / 2)) / RENDER_WIDTH
        cy = float(raw.get("object_center_y", RENDER_HEIGHT / 2)) / RENDER_HEIGHT
        half = None
        if cx < 0.4:
            half = "left"
        elif cx > 0.6:
            half = "right"
        return {"text": goal_prompt(goal_vector(raw, NL_GOAL_KEYS), NL_GOAL_KEYS, crop=raw),
                "target_point": (cx, cy), "requested_half": half}

    @staticmethod
    def _subject_front_az(ep):
        """World azimuth of the subject's front, recovered from the manifest.

        The render server reads this from configs/policy/facing_map_final.json, but our object
        stems only match it on 50/64 manifest episodes. The manifest publishes `bearing` for the
        goal on all 64, and bearing = (front_az - az) mod 360, so front_az = (bearing + az) mod
        360 is exact wherever the goal pose is known. Same information the VLAs' prompts already
        carry -- without it "from the subject's back-right" cannot be placed in the world at all.
        """
        gf = ep.goal_frame
        b = (gf.raw or {}).get("subject_bearing_deg")
        if b is None:
            return None
        geo = pose_to_geometry(gf.camera_position, gf.camera_forward, gf.camera_up,
                               gf.subject_center, gf.subject_height)
        return (float(b) + math.degrees(geo["az"])) % 360.0

    def search(self, rend, ep, g):
        """Drive the renderer directly; returns (final_pose, trace, info)."""
        scout = rend.scene_scout() or {}
        fa = self._subject_front_az(ep)
        if fa is not None:
            subj = dict(scout.get("subject") or {})
            subj.setdefault("front_azimuth_deg", round(fa, 1))
            subj.setdefault("front_vector_world",
                            [round(math.cos(math.radians(fa)), 3),
                             round(math.sin(math.radians(fa)), 3), 0.0])
            subj.setdefault("bearing_convention",
                            "bearing B = angle around the subject measured from its front. The subject's front points along front_vector_world. IMPORTANT: to shoot at bearing B, place the CAMERA at world azimuth (front_azimuth_deg - B + 180) mod 360 from the subject centre, i.e. camera = subject_centre + distance * [cos(theta), sin(theta), height_offset] with theta = radians((front_azimuth_deg - B + 180) mod 360), then look_at the subject. (front_azimuth_deg - B) alone is the direction FROM the camera TO the subject, which is the opposite side -- using it places the camera 180 deg wrong.")
            scout = {**scout, "subject": subj}
        t = {"n": 0}

        def render(position, look_at):
            pos, fwd, up = self._pose(position, look_at)
            t["n"] += 1
            img, achieved = rend.render(t["n"], pos, fwd, up, True)
            achieved = achieved or {}
            return img, achieved, bool(achieved.get("in_frame", False))

        best, info = self.agent.search(
            render, scout, g["text"], requested_half=g["requested_half"],
            target_point=g["target_point"], resolution=(int(RENDER_WIDTH), int(RENDER_HEIGHT)))
        if best is None:
            return None, info
        return self._pose(best["position"], best["look_at"]), info


class UNICComposition(_FiveDPolicy):
    """UNIC (ICCV'23 "Beyond Image Borders") composition model driven in 3D.

    The published pretrained model, used as-is, via the reviewed wrapper in
    `src/policy/unic/` (see its REFERENCES.md): UNIC recommends a composition box for the
    current frame, and `UNICPolicy` maps that box to a camera-local 5D move (box centre ->
    pan, box width -> dolly) using the real render FOV.

    It is GOAL-AGNOSTIC by construction: a 2D composition model has no notion of the
    requested shot, so it moves toward whatever it considers well-composed. That is exactly
    the point of the row -- it isolates how much of this task is generic composition versus
    goal-following, and it is the "reactive, no previsualization" end of the axis.
    """

    def __init__(self, checkpoint="weights/unic/unic_pretrained.pth", device="cuda"):
        from src.policy.unic.model import UNICModel
        from src.policy.unic.policy import UNICPolicy
        self.inner = UNICPolicy(UNICModel.load(checkpoint, device=device),
                                max_translation_m=self.max_trans, max_angle_deg=self.max_ang_deg)

    def goal(self, ep):
        return None                                  # goal-agnostic by construction

    def act(self, image_path, goal):
        from PIL import Image
        with Image.open(image_path) as im:
            a5, meta = self.inner.act(im.convert("RGB"))
        rec = meta.get("recommendation", {})
        return a5, 0.0, {k: round(float(v), 3) for k, v in rec.items()}


def _parse_json_obj(text):
    """Pull the first JSON object out of a model reply (handles ``` fences and prose)."""
    import re
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, *(m.group(0) for m in re.finditer(r"\{.*?\}", t, re.S))):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return None


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

    def scene_scout(self):
        """The one-shot geometric scene summary the server writes at load (PhotoFlow Director)."""
        f = self.ctl / "scene.json"
        return json.loads(f.read_text()) if f.exists() else None

    def render(self, t, pos, fwd, up, want_profile=False):
        """Returns the image path, or (image, achieved_profile) when want_profile."""
        req = {"t": t, "pose": {"pos": list(map(float, pos)), "forward": list(map(float, fwd)),
                                "up": list(map(float, up))}}
        (self.ctl / "req.json.tmp").write_text(json.dumps(req))
        (self.ctl / "req.json.tmp").rename(self.ctl / "req.json")
        resp = self.ctl / f"resp_{t:03d}.json"
        for _ in range(12000):
            if resp.exists():
                r = json.loads(resp.read_text())
                return (r["image"], r.get("achieved")) if want_profile else r["image"]
            time.sleep(0.05)
        raise RuntimeError(f"render timeout t={t}")

    def close(self):
        try:
            (self.ctl / "stop.flag").write_text("ok"); time.sleep(2); self.proc.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------- episodes
def episodes_from_manifest(path, roots, extra_chunks):
    """Shared-episode mode: V12's configs/eval_episodes_v1.json.

    Poses-only (option 1 of docs/shared_eval_protocol.md): start/goal poses, subject
    geometry and n_chunks all come from the manifest, so this does NOT depend on our
    window enumeration matching theirs. Only `data_json` is rebuilt against our own root
    (their absolute paths are on their NAS).

    `d_start` is the contract checksum: recompute it from the manifest's own poses and
    refuse to run on a mismatch, because an episode set that silently differs makes every
    downstream number incomparable.
    """
    m = json.loads(Path(path).read_text())
    eps, missing, bad = [], [], []
    for e in m["episodes"]:
        dj = next((Path(r) / e["placement"] / "data.json" for r in roots
                   if (Path(r) / e["placement"] / "data.json").exists()), None)
        if dj is None:
            missing.append(e["placement"]); continue
        # goal_raw carries the WORLD azimuth; subject_bearing_deg is normally derived by our
        # loader via the facing map, which the manifest path bypasses. Take the manifest's
        # own `bearing` (verified to match our facing map on all 64 episodes) so both the DP
        # goal vector and the NL prompt see the key they expect instead of NaN.
        raw = dict(e["goal_raw"])
        raw.setdefault("subject_bearing_deg", float(e["bearing"]))
        gf = SimpleNamespace(raw=raw,
                             camera_position=e["goal_pose"]["position"],
                             camera_forward=e["goal_pose"]["forward"],
                             camera_up=e["goal_pose"]["up"],
                             subject_center=e["subject_center"],
                             subject_height=e["subject_height"])
        st = SimpleNamespace(camera_position=e["start_pose"]["position"],
                             camera_forward=e["start_pose"]["forward"],
                             camera_up=e["start_pose"]["up"])
        d = geometry_distance(st.camera_position, st.camera_forward, st.camera_up, gf)
        if abs(d - e["d_start"]) >= 1e-6:
            bad.append((e["key"], d, e["d_start"]))
        # NB no goal_image/start_image here. The manifest's image paths live on V12's NAS, and
        # remapping them into our renders/ by BASENAME silently returns the wrong frame: the
        # filenames line up (pair_00_frame_29.jpg, 32 frames) but our trajectory_32f is a
        # different render pass, so frame 29 is a blank wall while the manifest's goal pose
        # looks at the subject. The d_start checksum cannot catch this -- it is recomputed from
        # the manifest's own poses and never touches our renders. Image-conditioned policies get
        # the goal pose RENDERED LIVE instead (see `needs_goal_render` in the rollout loop).
        eps.append(SimpleNamespace(key=e["key"], name=e["placement"], scene=e["scene"],
                                   data_json=dj, start=st, goal_frame=gf,
                                   n_chunks=int(e["n_chunks"]) + extra_chunks))
    if bad:
        raise SystemExit(f"d_start CHECKSUM FAILED on {len(bad)} episode(s), e.g. {bad[:2]} -- "
                         "the two repos are not rolling out the same shots; refusing to run")
    if missing:
        print(f"WARNING: {len(missing)} manifest placements absent locally: {missing[:3]}", flush=True)
    print(f"manifest {path}: {len(eps)}/{len(m['episodes'])} episodes, d_start checksum OK", flush=True)
    return eps


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
    ap.add_argument("--policy", required=True,
                    choices=["dp", "pi05", "groot", "gpt5", "unic", "photoflow"])
    ap.add_argument("--render-budget", type=int, default=None, dest="render_budget",
                    help="photoflow: cap total preview renders. Default = rounds*candidates = 24 "
                         "(the paper's budget). Set it to n_chunks for a render-matched row.")
    ap.add_argument("--model", default="gpt-5", help="gateway model id for --policy gpt5")
    ap.add_argument("--feedback", type=int, default=0,
                    help="gpt5: also give the agent its MEASURED current profile and the remaining "
                         "error (PhotoFlow's Reviewer signal). Off = matched to the VLAs, which "
                         "see only image+goal; on = strictly more information, reported separately.")
    ap.add_argument("--goal-mode", default="text", choices=["text", "image"], dest="goal_mode",
                    help="gpt5 only: condition on the NL shot brief (text, comparable to the "
                         "VLAs) or on the goal frame itself (image, strictly more information)")
    ap.add_argument("--checkpoint", default=None,
                    help="not used by the training-free rows (gpt5, unic)")
    ap.add_argument("--config", default=None, help="DP training config (required for --policy dp)")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="val = held-out (generalization); train = the control V12 added in be8f72f")
    ap.add_argument("--val-scenes", default="configs/policy/val_scenes.json", dest="val_scenes")
    ap.add_argument("--roots", nargs="+", default=["data/trajectories_full"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--episodes-from", default=None, dest="episodes_from",
                    help="shared episode manifest (configs/eval_episodes_v1.json). Fixes the "
                         "episode set AND the horizon, so --episodes/--split no longer apply.")
    ap.add_argument("--chunk-size", type=int, default=8, dest="chunk_size")
    ap.add_argument("--extra-chunks", type=int, default=0, dest="extra_chunks")
    ap.add_argument("--shoot-threshold", type=float, default=0.5, dest="shoot_threshold")
    ap.add_argument("--stop-on-shoot", type=int, default=1, dest="stop_on_shoot")
    ap.add_argument("--blender", default="blender/blender")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    if a.policy == "dp":
        pol = DPPolicy(a.checkpoint, a.config, dev, chunk=a.chunk_size)
    elif a.policy in ("pi05", "groot"):
        pol = LeRobotPolicy(a.policy, a.checkpoint, dev)
    elif a.policy == "gpt5":
        pol = GPT5Agent(a.model, goal_mode=a.goal_mode, feedback=bool(a.feedback))
    elif a.policy == "photoflow":
        pol = PhotoFlowPolicy(a.model, render_budget=a.render_budget)
    else:
        pol = UNICComposition(a.checkpoint or "weights/unic/unic_pretrained.pth")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.episodes_from:
        episodes = episodes_from_manifest(a.episodes_from, a.roots, a.extra_chunks)
        print(f"{a.policy} | SHARED MANIFEST | {len(episodes)} episodes", flush=True)
    else:
        episodes = []
        for name, dj in select_placements(a.split, a.val_scenes, a.roots, a.episodes):
            try:
                w = next(iter(iter_goal_start_windows(str(dj), chunk_size=a.chunk_size)))
            except StopIteration:
                continue
            delta = abs(w.goal_frame.frame_idx - w.start_frame_idx)
            episodes.append(SimpleNamespace(
                key=None, name=name, scene=name.split("__")[0], data_json=dj,
                start=w.start, goal_frame=w.goal_frame,
                n_chunks=int(math.ceil(delta / a.chunk_size)) + a.extra_chunks))
        print(f"{a.policy} | split={a.split} | {len(episodes)} episodes", flush=True)

    results = []
    for i, ep in enumerate(episodes):
        name, dj, n_chunks = ep.name, ep.data_json, ep.n_chunks
        pos = np.asarray(ep.start.camera_position, np.float32)
        fwd = np.asarray(ep.start.camera_forward, np.float32)
        up = np.asarray(ep.start.camera_up, np.float32)
        d0 = geometry_distance(pos, fwd, up, ep.goal_frame)
        trace, declared, notes = [d0], None, []
        try:
            rend = Renderer(dj, V6_DIR / f"{name}.json", out / (ep.key or name), a.blender)
        except Exception as e:
            print(f"  [{i}] {name[:40]}: renderer failed ({e})", flush=True); continue
        try:
            if getattr(pol, "needs_goal_render", False):
                gf = ep.goal_frame
                g = pol.goal(ep, goal_image=rend.render(
                    GOAL_RENDER_T, np.asarray(gf.camera_position, np.float32),
                    np.asarray(gf.camera_forward, np.float32),
                    np.asarray(gf.camera_up, np.float32)))
            else:
                g = pol.goal(ep)
        except Exception as e:
            rend.close()
            print(f"  [{i}] {name[:40]}: goal setup failed ({e})", flush=True); continue
        if getattr(pol, "search_space", "incremental") == "global":
            # A global search that finds nothing leaves the trace at d_start, i.e. it scores
            # exactly 0 -- the same convention a no-op gets everywhere else in this harness.
            info = {}
            try:
                final, info = pol.search(rend, ep, g)
                if final is not None:
                    pos, fwd, up = final
                    trace.append(geometry_distance(pos, fwd, up, ep.goal_frame))
            except Exception as e:
                info = {**info, "error": f"{type(e).__name__}: {e}"[:200]}
                print(f"  [{i}] {name[:40]}: search failed ({info['error']})", flush=True)
            finally:
                rend.close()
            notes.append({k: v for k, v in info.items() if k != "log"})
            epdir = out / (ep.key or name)
            epdir.mkdir(parents=True, exist_ok=True)
            (epdir / "photoflow_log.json").write_text(json.dumps(info, indent=1))
            d1 = trace[-1]
            dbest = min(trace)
            results.append({"key": ep.key, "placement": name, "scene": ep.scene,
                            "n_chunks": n_chunks, "renders": info.get("renders"),
                            "d_start": d0, "d_end": d1, "d_best": dbest,
                            "improvement": d0 - d1, "best_improvement": d0 - dbest,
                            "trace": trace, "declared": None, "notes": notes})
            print(f"  [{i}] {name[:40]:40s} d {d0:.3f}->{d1:.3f} (best {dbest:.3f}) "
                  f"improve {d0 - d1:+.3f} renders {info.get('renders')}", flush=True)
            continue

        try:
            want = getattr(pol, "wants_profile", False)
            img, prof = (rend.render(0, pos, fwd, up, True) if want
                         else (rend.render(0, pos, fwd, up), None))
            for c in range(n_chunks):
                if getattr(pol, "action_space", "9d") == "5d":
                    # one interpretable 5D move per observation (see _FiveDPolicy)
                    act5, shoot, meta = (pol.act(img, g, prof) if want else pol.act(img, g))
                    notes.append(meta)
                    if declared is None and shoot > a.shoot_threshold:
                        declared = c
                        if a.stop_on_shoot:
                            break
                    pos, fwd, up = apply_action_5d(pos, fwd, up, act5)
                else:
                    chunk = pol.act(img, g)
                    shoot = float(np.max(chunk[:, POSE_DIM])) if chunk.shape[1] > POSE_DIM else 0.0
                    if declared is None and shoot > a.shoot_threshold:
                        declared = c
                        if a.stop_on_shoot:
                            break                   # stop BEFORE executing (V12 semantics)
                    for step in chunk[:, :POSE_DIM]:
                        pos, fwd, up = apply_action_9d(pos, fwd, up, step)
                img, prof = (rend.render(c + 1, pos, fwd, up, True) if want
                             else (rend.render(c + 1, pos, fwd, up), None))
                trace.append(geometry_distance(pos, fwd, up, ep.goal_frame))
        finally:
            rend.close()
        d1, dbest = trace[-1], float(min(trace))
        results.append({"key": ep.key, "placement": name, "scene": ep.scene, "n_chunks": n_chunks,
                        "declared_stop": declared, "d_start": d0, "d_end": d1, "d_best": dbest,
                        "improvement": d0 - d1, "best_improvement": d0 - dbest, "trace": trace,
                        **({"notes": notes} if notes else {})})
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
        # success@tau: `improvement` says the camera got closer, not that the requested
        # photograph was taken. V12's calibrate_shot_threshold puts ~66% of real frame pairs
        # under d<0.10 on the same six cinematography words, ~0% past 0.60.
        "success_at": {f"{t}": {"end": float(np.mean([r["d_end"] <= t for r in results])),
                                "best": float(np.mean([r["d_best"] <= t for r in results]))}
                       for t in (0.05, 0.10, 0.20)} if results else None,
        "episodes_from": a.episodes_from,
        "model": a.model if a.policy == "gpt5" else None,
        "goal_mode": a.goal_mode if a.policy == "gpt5" else None,
        "feedback": bool(a.feedback) if a.policy == "gpt5" else None,
        "render_budget": (pol.agent.render_budget if a.policy == "photoflow" else None),
        # PhotoFlow's counters live on the inner agent, not the adapter -- reading only
        # `pol.calls` reported 0 calls for a run that actually made 21,588.
        "api_calls": getattr(pol, "calls", None) or getattr(getattr(pol, "agent", None), "calls", None),
        "api_failures": (getattr(pol, "failures", None)
                         or getattr(getattr(pol, "agent", None), "failures", None)),
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
