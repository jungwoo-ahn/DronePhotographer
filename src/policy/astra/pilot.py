"""Astra-style VLM drone pilot (jirl-upenn.github.io/blog/astra-as-a-drone-pilot.html).

The blog flies a drone with "GPT-6 Astra": a single forward-facing RGB frame plus a text
prompt go to the model, which replies with a *velocity command* and a free-text scratchpad.
Every frame, command and scratchpad is appended to a growing context and resent, so the pilot
accumulates its own history. The command is held constant until the next query (velocity mode,
2 Hz), and a `status` field of "Complete" ends the run.

Faithful here:
  - the action vector `{vx, vy, vz, yaw_rate_deg_s, status, scratchpad}` and its signs
    (vy positive = LEFT, vz positive = climb, yaw_rate positive = left);
  - the published limits |vx|<5, |vy|<1.5, |vz|<1, |yaw_rate|<120 deg/s, held for `dt`;
  - 2 Hz (dt = 0.5 s), low reasoning effort, no explicit CoT;
  - accumulated context: every observation, command and scratchpad is resent;
  - `status: "Complete"` is the pilot's own termination decision.

Two documented departures, both forced by the task rather than by convenience:
  1. **Gimbal pitch.** The blog's velocity mode keeps the camera level — its task is to reach a
     position ("within 1 m of the target"), so aim never matters. Ours is a *photography* task
     scored partly on where the subject sits in frame, and a level-locked camera can never fix
     the aim term. `pitch_rate_deg_s` is added to the action vector; everything else is the
     blog's. A real camera drone has a gimbal, so this is the honest analogue.
  2. **Goal conditioning.** The blog describes targets in words only ("a man in a red
     short-sleeved shirt"). Here the goal is the shot to achieve, given either as the NL shot
     brief or as a reference image, so the pilot is comparable to the other goal-conditioned rows.
"""

from __future__ import annotations

import json
import math

import numpy as np

from src.policy.common.action_repr import POSE_DIM_5D, apply_action_5d
from src.utils.rotation_utils import make_camera_basis_from_forward_up

HZ = 2.0
DT = 1.0 / HZ                       # command held constant between queries
MAX_VX, MAX_VY, MAX_VZ = 5.0, 1.5, 1.0          # m/s, blog's velocity-mode limits
MAX_YAW_RATE = 120.0                            # deg/s
MAX_PITCH_RATE = 60.0                           # deg/s, the added gimbal axis
WORLD_UP = np.array([0.0, 0.0, 1.0])
FIELDS = ("vx", "vy", "vz", "yaw_rate_deg_s", "pitch_rate_deg_s")


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
    """Hold `cmd` for `dt` and return the resulting (position, forward, up).

    vx is along the camera's forward *levelled to horizontal*, vy is horizontal-left, and vz is
    world up — so the translation is built in the world frame, then expressed in the camera
    basis and handed to `apply_action_5d`, which already applies yaw about world up and pitch
    about the local right axis. Astra's yaw_rate is LEFT-positive, which is the same convention
    as this repr's `dyaw`, so it passes through with no sign flip (pinned by a test, because
    every other yaw in this codebase needed one).
    """
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd_h = np.array([forward[0], forward[1], 0.0])
    n = float(np.linalg.norm(fwd_h))
    if n < 1e-6:                                   # looking straight up/down: fall back to up
        fwd_h = np.array([up[0], up[1], 0.0])
        n = float(np.linalg.norm(fwd_h))
        if n < 1e-6:
            fwd_h, n = np.array([1.0, 0.0, 0.0]), 1.0
    fwd_h /= n
    left_h = np.cross(WORLD_UP, fwd_h)             # horizontal left of the heading

    delta_world = (float(cmd["vx"]) * dt * fwd_h
                   + float(cmd["vy"]) * dt * left_h
                   + float(cmd["vz"]) * dt * WORLD_UP)

    basis = make_camera_basis_from_forward_up(forward.astype(np.float32), up.astype(np.float32))
    delta_local = basis.T.astype(np.float64) @ delta_world      # columns are (right, up, fwd)

    a = np.zeros(POSE_DIM_5D, dtype=np.float32)
    a[0], a[1], a[2] = delta_local[0], delta_local[1], delta_local[2]
    a[3] = math.radians(float(cmd["yaw_rate_deg_s"]) * dt)      # left-positive both sides
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


class AstraPilot:
    """Accumulating-context velocity pilot. One query per rendered frame."""

    def __init__(self, vlm, goal_mode="text", dt=DT, max_context_frames=12):
        self.vlm = vlm
        self.goal_mode = goal_mode
        self.dt = float(dt)
        # The blog resends every frame; context cost grows ~quadratically, so keep a window of
        # the most recent frames and let the scratchpad carry anything older.
        self.max_context_frames = int(max_context_frames)
        self.reset()

    def reset(self):
        self.history: list[dict] = []
        self.scratchpad = ""
        self.calls = self.failures = 0

    def _messages(self, goal, image_path):
        """Rebuild the accumulated context: goal, then every past (frame, command), then now."""
        content = []
        if goal.get("mode") == "image":
            content += [{"type": "text", "text": "REQUESTED PHOTOGRAPH (reproduce this view):"},
                        {"type": "image", "path": goal["image"]}]
        else:
            content.append({"type": "text", "text": f"REQUESTED PHOTOGRAPH: {goal['text']}"})

        window = self.history[-self.max_context_frames:]
        if len(self.history) > len(window):
            content.append({"type": "text",
                            "text": f"({len(self.history) - len(window)} earlier steps omitted; "
                                    f"your running notes carry them.)"})
        for i, h in enumerate(window):
            content.append({"type": "text", "text": f"--- step {h['step']}, what you saw:"})
            content.append({"type": "image", "path": h["image"]})
            content.append({"type": "text",
                            "text": "you commanded " + json.dumps(
                                {k: h["cmd"][k] for k in FIELDS})
                                    + f"\nyour note: {h['cmd']['scratchpad']}"})
        content.append({"type": "text", "text": f"--- step {len(self.history)}, NOW:"})
        content.append({"type": "image", "path": image_path})
        if self.scratchpad:
            content.append({"type": "text", "text": f"your latest note: {self.scratchpad}"})
        content.append({"type": "text", "text": "Command for the next "
                                                f"{self.dt:.1f} s? Strict JSON only."})
        return content

    def act(self, image_path, goal):
        """One query. Returns (command dict, done flag)."""
        self.calls += 1
        d = self.vlm.ask_json(system=SYSTEM, content=self._messages(goal, image_path))
        if not isinstance(d, dict):
            self.failures += 1
            cmd = clamp_command({})                     # a dropped reply = hold still, keep flying
            cmd["scratchpad"] = self.scratchpad
            cmd["parse_failed"] = True
        else:
            cmd = clamp_command(d)
            self.scratchpad = cmd["scratchpad"] or self.scratchpad
        self.history.append({"step": len(self.history), "image": image_path, "cmd": cmd})
        return cmd, cmd["status"] == "Complete"


__all__ = ["AstraPilot", "apply_velocity", "clamp_command", "SYSTEM",
           "HZ", "DT", "MAX_VX", "MAX_VY", "MAX_VZ", "MAX_YAW_RATE", "MAX_PITCH_RATE", "FIELDS"]
