"""Astra-style VLM drone pilot, vendored from the shared repo for the V12 benchmark harness.

Source: jungwoo-ahn/DronePhotographer @ origin/main 86836188 (PR #37, jooyeolyun),
`src/policy/astra/pilot.py` and `AstraPolicy` / `GatewayVLM` in
`scripts/closed_loop_eval_baselines.py`. After jirl-upenn.github.io/blog/astra-as-a-drone-pilot.html.

What is the SAME as the source (do not change without saying so in the paper):
  * the system prompt, the action vector {vx, vy, vz, yaw_rate_deg_s, pitch_rate_deg_s, status,
    scratchpad}, its signs and clamps, dt = 0.5 s held per query;
  * accumulated context: goal, then the last 12 (frame, command, note) triples, then the current
    frame -- rebuilt and resent every query;
  * `status: "Complete"` ends the run BEFORE that command is executed;
  * gpt-6-astra, reasoning_effort "low", temperature 0.2, max_tokens 8192 (doubled once when the
    reply is empty), first JSON object of the reply; a reply that cannot be parsed = hover;
  * the text goal is the source repo's NL brief (`src.baselines.goal_text.goal_prompt`), the
    image goal the goal frame itself.

What differs, and why:
  * frames (and the goal image) are sent at 296x222: 4:3 at the pixel count of our 256x256
    policy input (65.7k vs 65.5k px). gpt-6-astra bills images at ~1 token per 32x32 patch
    (measured 2026-11-06: 1024x768 -> 769 tok, 296x222 -> 71, 256x256 -> 65), so this is the
    same visual budget our model gets, at ~1/11 the image tokens of the full render. The aspect
    is kept rather than squashed because the prompt states the FOV (29.9 x 22.6 deg);
  * with `goal_framing="cross_reference"` (image goals from another scene and person) the goal
    intro and the two system-prompt sentences that say "reproduce / matches this picture" are
    reworded, since that picture can never be matched; the source never ran this condition;
  * transport errors (429 / 5xx / timeouts) are retried with backoff before being treated as a
    dropped reply. The source turned every exception into an immediate hover, which scores
    gateway trouble as policy failure. Every attempt is logged, so the difference is auditable;
  * every call is timed and its token usage, cost and request id recorded (`calls` log), for the
    paper's inference-cost table.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from src.common.action_repr import ACTION_DIM_5D, apply_action_5d
from src.utils.rotation_utils import make_camera_basis_from_forward_up

HZ = 2.0
DT = 1.0 / HZ
MAX_VX, MAX_VY, MAX_VZ = 5.0, 1.5, 1.0
MAX_YAW_RATE = 120.0
MAX_PITCH_RATE = 60.0
WORLD_UP = np.array([0.0, 0.0, 1.0])
FIELDS = ("vx", "vy", "vz", "yaw_rate_deg_s", "pitch_rate_deg_s")

GATEWAY = "https://gw.letsur.ai/v1"
# Gateway price list (GET /v1/models, 2026-11-06), USD per token.
PRICES = {"gpt-6-astra": (10e-6, 50e-6), "gpt-5": (1.25e-6, 10e-6)}
KEY_FILE = os.path.expanduser("~/.config/letsur/key")   # a bare key, mode 600
ENV_FILE = "/home/nas_main/jungwooahn/projects/DronePhotographerV12/.env"


def clamp_command(d: dict) -> dict:
    """Parse + clamp a reply into the action vector. Missing or junk fields become 0."""
    def f(key, lim):
        try:
            return float(np.clip(float(d.get(key, 0.0) or 0.0), -lim, lim))
        except (TypeError, ValueError):
            return 0.0
    return {"vx": f("vx", MAX_VX), "vy": f("vy", MAX_VY), "vz": f("vz", MAX_VZ),
            "yaw_rate_deg_s": f("yaw_rate_deg_s", MAX_YAW_RATE),
            "pitch_rate_deg_s": f("pitch_rate_deg_s", MAX_PITCH_RATE),
            "status": "Complete" if str(d.get("status", "")).strip().lower() == "complete"
                      else "Searching",
            "scratchpad": str(d.get("scratchpad", ""))[:600]}


def apply_velocity(position, forward, up, cmd: dict, dt: float = DT):
    """Hold `cmd` for `dt`; return (position, forward, up). Identical to the source.

    vx is along the heading levelled to horizontal, vy horizontal-left, vz world up; the world
    translation is expressed in the camera basis and handed to `apply_action_5d`, which yaws
    about world up (left-positive, same as Astra -- no sign flip) and pitches about local right.
    """
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd_h = np.array([forward[0], forward[1], 0.0])
    n = float(np.linalg.norm(fwd_h))
    if n < 1e-6:
        fwd_h = np.array([up[0], up[1], 0.0])
        n = float(np.linalg.norm(fwd_h))
        if n < 1e-6:
            fwd_h, n = np.array([1.0, 0.0, 0.0]), 1.0
    fwd_h /= n
    left_h = np.cross(WORLD_UP, fwd_h)
    delta_world = (float(cmd["vx"]) * dt * fwd_h + float(cmd["vy"]) * dt * left_h
                   + float(cmd["vz"]) * dt * WORLD_UP)
    basis = make_camera_basis_from_forward_up(forward.astype(np.float32), up.astype(np.float32))
    delta_local = basis.T.astype(np.float64) @ delta_world
    a = np.zeros(ACTION_DIM_5D, dtype=np.float32)
    a[0], a[1], a[2] = delta_local[0], delta_local[1], delta_local[2]
    a[3] = math.radians(float(cmd["yaw_rate_deg_s"]) * dt)
    a[4] = math.radians(float(cmd["pitch_rate_deg_s"]) * dt)
    return apply_action_5d(position, forward, up, a)


SYSTEM = (
    "You are piloting a camera drone in a simulated 3D scene. You see the drone's forward "
    "camera. Your job is to fly so the camera ends up taking the REQUESTED PHOTOGRAPH of the "
    "subject.\n"
    "Each reply is strict JSON only, no prose:\n"
    '{"vx":<f>,"vy":<f>,"vz":<f>,"yaw_rate_deg_s":<f>,"pitch_rate_deg_s":<f>,'
    '"status":"Searching"|"Complete","scratchpad":"<free text>"}\n'
    f"vx = forward m/s along the heading, levelled horizontal (|vx| < {MAX_VX}). "
    f"vy = sideways m/s, POSITIVE IS LEFT (|vy| < {MAX_VY}). "
    f"vz = climb m/s, positive is up (|vz| < {MAX_VZ}). "
    f"yaw_rate_deg_s turns the drone, POSITIVE IS LEFT (|yaw| < {MAX_YAW_RATE}). "
    f"pitch_rate_deg_s tilts the camera gimbal, positive is up (|pitch| < {MAX_PITCH_RATE}).\n"
    f"Each command is held for {DT:.1f} s and then you are asked again, so think of each reply "
    "as half a second of flight.\n"
    "The lens is narrow: horizontal field of view 29.9 deg, vertical 22.6 deg, and the subject "
    "is usually about 5 m away. Sideways motion of much more than ~1.3 m without turning back "
    "toward the subject will push it out of frame, and once it is gone you are flying blind. To "
    "change which SIDE of the subject you are shooting from you must fly around it: use vy to "
    "arc sideways and yaw_rate_deg_s at the same time to keep it centred.\n"
    "scratchpad is yours: carry notes forward about where the subject is, which way you are "
    "circling, and what you have already tried. You will see your own earlier scratchpads.\n"
    'status is "Searching" while you are still flying, and "Complete" the moment the current '
    "view matches the requested photograph — that ends the run and takes the picture, so say it "
    "when the shot is right, and not before."
)


# Cross-reference goals: the picture is the same SHOT from another scene with another person, so it
# can never be matched. Only the sentences that say what the picture is and when to stop change.
_XREF_REPLACEMENTS = (
    ("taking the REQUESTED PHOTOGRAPH of the subject.",
     "taking a photograph of your subject composed like the REFERENCE PHOTOGRAPH."),
    ('"Complete" the moment the current view matches the requested photograph',
     '"Complete" the moment your subject is framed the way the reference frames its subject'),
)
SYSTEM_CROSS_REFERENCE = SYSTEM
for _old, _new in _XREF_REPLACEMENTS:
    assert _old in SYSTEM_CROSS_REFERENCE, _old
    SYSTEM_CROSS_REFERENCE = SYSTEM_CROSS_REFERENCE.replace(_old, _new)
GOAL_INTRO = {
    "same_view": "REQUESTED PHOTOGRAPH (reproduce this view):",
    "cross_reference": ("REFERENCE PHOTOGRAPH, taken in a different place with a different person. "
                        "Frame YOUR subject the same way: same viewing side and angle, same size and "
                        "position in the frame:"),
}


def parse_json_obj(text):
    """First JSON object in a reply (handles ``` fences and prose). Same as the source."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for cand in (t, *(m.group(0) for m in re.finditer(r"\{.*?\}", t, re.S))):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            continue
    return None


def load_key() -> str:
    key = os.environ.get("LETSUR_API_KEY") or os.environ.get("GATEWAY_API_KEY")
    if not key and os.access(ENV_FILE, os.R_OK):
        m = re.search(r"(?:LETSUR|GATEWAY)_API_KEY\s*=\s*['\"]?([^'\"\s]+)", Path(ENV_FILE).read_text())
        key = m.group(1) if m else None
    if not key and Path(KEY_FILE).exists():
        key = Path(KEY_FILE).read_text().strip() or None
    if not key:
        raise SystemExit(f"no LETSUR_API_KEY in env, {KEY_FILE} or {ENV_FILE}")
    return key


class GatewayVLM:
    """`ask_json` over the LetSur gateway, with per-call timing / usage / cost records."""

    def __init__(self, model="gpt-6-astra", *, reasoning="low", temperature=0.2, max_tokens=8192,
                 timeout_s=180.0, max_attempts=6, post=None):
        self.model, self.reasoning, self.temperature = model, reasoning, temperature
        self.max_tokens, self.timeout_s, self.max_attempts = int(max_tokens), timeout_s, max_attempts
        self._key = None if post else load_key()
        self._post = post or self._http_post
        self.calls: list[dict] = []          # one record per ask_json, appended in order

    @staticmethod
    def _img(path):
        return {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(Path(path).read_bytes()).decode()}"}}

    def _http_post(self, body: dict) -> tuple[int, dict, dict]:
        req = urllib.request.Request(f"{GATEWAY}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self._key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.load(r), dict(r.headers)
        except urllib.error.HTTPError as e:
            try:
                payload = json.load(e)
            except Exception:  # noqa: BLE001
                payload = {}
            return e.code, payload, dict(e.headers or {})

    def ask_json(self, system, content):
        blocks = [b if b["type"] != "image" else self._img(b["path"]) for b in content]
        n_images = sum(b["type"] == "image" for b in content)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": blocks}]
        rec = {"t_wall": time.time(), "n_images": n_images, "attempts": [], "parsed": False}
        t_call = time.perf_counter()
        reply = None
        for budget in (self.max_tokens, self.max_tokens * 2):
            body = {"model": self.model, "max_tokens": budget, "temperature": self.temperature,
                    "messages": msgs}
            if self.reasoning:
                body["reasoning_effort"] = self.reasoning
            txt, ok = None, False
            for attempt in range(self.max_attempts):
                t0 = time.perf_counter()
                try:
                    status, payload, headers = self._post(body)
                    err = None
                except Exception as exc:  # noqa: BLE001  (timeouts, resets)
                    status, payload, headers, err = None, {}, {}, f"{type(exc).__name__}: {exc}"
                lat = time.perf_counter() - t0
                u = payload.get("usage") or {}
                det = u.get("completion_tokens_details") or {}
                rec["attempts"].append({
                    "budget": budget, "status": status, "latency_s": round(lat, 4), "error": err,
                    "request_id": headers.get("x-request-id") or headers.get("X-Request-Id")
                                  or payload.get("id"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": det.get("reasoning_tokens"),
                    "finish_reason": ((payload.get("choices") or [{}])[0]).get("finish_reason")})
                if status == 200:
                    txt = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                    ok = True
                    break
                retryable = status is None or status == 429 or status >= 500
                if not retryable:
                    break
                time.sleep(min(60.0, 2.0 * 2 ** attempt))
            if not ok:
                break                         # transport gave up: dropped reply
            if txt.strip():
                reply = parse_json_obj(txt)
                rec["raw_reply"] = txt[:2000]
                break
        rec["latency_s"] = round(time.perf_counter() - t_call, 4)     # incl. retries/backoff
        ok_atts = [a for a in rec["attempts"] if a["status"] == 200]
        rec["api_latency_s"] = round(sum(a["latency_s"] for a in ok_atts), 4)
        rec["prompt_tokens"] = sum(a["prompt_tokens"] or 0 for a in rec["attempts"])
        rec["completion_tokens"] = sum(a["completion_tokens"] or 0 for a in rec["attempts"])
        rec["reasoning_tokens"] = sum(a["reasoning_tokens"] or 0 for a in rec["attempts"])
        pin, pout = PRICES.get(self.model, (float("nan"), float("nan")))
        rec["cost_usd"] = rec["prompt_tokens"] * pin + rec["completion_tokens"] * pout
        rec["parsed"] = isinstance(reply, dict)
        self.calls.append(rec)
        return reply


class AstraPilot:
    """Accumulating-context velocity pilot. One query per rendered frame. Same as the source."""

    def __init__(self, vlm, dt=DT, max_context_frames=12, goal_framing="same_view"):
        self.vlm, self.dt, self.max_context_frames = vlm, float(dt), int(max_context_frames)
        self.goal_intro = GOAL_INTRO[goal_framing]
        self.system = SYSTEM_CROSS_REFERENCE if goal_framing == "cross_reference" else SYSTEM
        self.calls = self.failures = 0
        self.reset()

    def reset(self):
        self.history: list[dict] = []
        self.scratchpad = ""

    def _messages(self, goal, image_path):
        content = []
        if goal.get("mode") == "image":
            content += [{"type": "text", "text": self.goal_intro},
                        {"type": "image", "path": goal["image"]}]
        else:
            content.append({"type": "text", "text": f"REQUESTED PHOTOGRAPH: {goal['text']}"})
        window = self.history[-self.max_context_frames:]
        if len(self.history) > len(window):
            content.append({"type": "text",
                            "text": f"({len(self.history) - len(window)} earlier steps omitted; "
                                    f"your running notes carry them.)"})
        for h in window:
            content.append({"type": "text", "text": f"--- step {h['step']}, what you saw:"})
            content.append({"type": "image", "path": h["image"]})
            content.append({"type": "text",
                            "text": "you commanded " + json.dumps({k: h["cmd"][k] for k in FIELDS})
                                    + f"\nyour note: {h['cmd']['scratchpad']}"})
        content.append({"type": "text", "text": f"--- step {len(self.history)}, NOW:"})
        content.append({"type": "image", "path": image_path})
        if self.scratchpad:
            content.append({"type": "text", "text": f"your latest note: {self.scratchpad}"})
        content.append({"type": "text", "text": f"Command for the next {self.dt:.1f} s? Strict JSON only."})
        return content

    def act(self, image_path, goal):
        self.calls += 1
        d = self.vlm.ask_json(system=self.system, content=self._messages(goal, image_path))
        if not isinstance(d, dict):
            self.failures += 1
            cmd = clamp_command({})
            cmd["scratchpad"] = self.scratchpad
            cmd["parse_failed"] = True
        else:
            cmd = clamp_command(d)
            self.scratchpad = cmd["scratchpad"] or self.scratchpad
        self.history.append({"step": len(self.history), "image": image_path, "cmd": cmd})
        return cmd, cmd["status"] == "Complete"


__all__ = ["AstraPilot", "GatewayVLM", "apply_velocity", "clamp_command", "parse_json_obj",
           "SYSTEM", "DT", "FIELDS", "PRICES"]
