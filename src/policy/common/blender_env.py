"""Blender-in-the-loop rollout environment (shared by the UNIC / AutoPhoto baselines).

A gym-like environment that moves the camera with our 5D action and re-renders the
view in Blender, so reactive / RL baselines can act on *actually rendered* frames
(not the analytic pose-proxy the trainable-policy evals use).

Design:
  - `Renderer` is a pluggable backend (`render(run_info, position, forward, up) ->
    PIL.Image`):
      * `SubprocessBlenderRenderer` — spawns `blender -b -P blender_render_pose.py`
        once per frame (the proven path; ~2-3s/frame scene-load overhead). Fine for
        UNIC's few-step eval.
      * `MockRenderer` — synthetic frame from the pose; for tests and for running the
        rollout logic without a Blender binary.
      * (AutoPhoto will add a persistent-worker backend that keeps one Blender process
        alive for fast RL rollouts — same interface.)
  - `BlenderRolloutEnv` holds the current pose, applies `apply_action_5d`, calls the
    renderer, and exposes the cheap analytic `pose_proxy_distance` for goal scoring.
    Full rendered shot-profile scoring is wired separately at eval time (reuses the
    existing detect+score pipeline) so the env stays light and detector-free.

NOTE: the Blender-backed path requires the `blender/blender` binary and a scene
`run_info.json`; it is exercised on the render machine. The pure-Python rollout
logic here is covered by tests using `MockRenderer`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.policy.common.action_repr import apply_action_5d


class Renderer(ABC):
    """Renders the view from a camera pose (world-frame position/forward/up)."""

    @abstractmethod
    def render(self, run_info_path: str, position: np.ndarray, forward: np.ndarray, up: np.ndarray):
        """Return a PIL.Image of the view from the given pose."""

    def close(self) -> None:  # backends that hold resources override this
        pass


class MockRenderer(Renderer):
    """Deterministic synthetic frame from the pose — no Blender. For tests / dry runs.

    The image content is a function of the pose so tests can assert the env actually
    re-renders after a move (different pose -> different pixels).
    """

    def __init__(self, size: tuple[int, int] = (64, 64)) -> None:
        self.size = size
        self.calls: list[dict] = []

    def render(self, run_info_path, position, forward, up):
        from PIL import Image

        self.calls.append({"position": np.asarray(position).tolist()})
        h, w = self.size
        seed = int(abs(float(np.sum(position) * 1000 + np.sum(forward) * 7))) % 256
        arr = np.full((h, w, 3), seed, dtype=np.uint8)
        return Image.fromarray(arr)


class SubprocessBlenderRenderer(Renderer):
    """Render one frame per `blender -b -P blender_render_pose.py` subprocess.

    Mirrors the invocation legacy `infer_mpc_blender.py` uses; correct but pays the
    scene-load cost each frame. Good enough for UNIC's short rollouts.
    """

    def __init__(
        self,
        blender_bin: str = "blender/blender",
        render_script: str = "scripts/blender_render_pose.py",
        repo_root: Optional[Path] = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.blender_bin = str((self.repo_root / blender_bin).resolve())
        self.render_script = str((self.repo_root / render_script).resolve())
        self.timeout_s = timeout_s
        if not Path(self.blender_bin).exists():
            raise FileNotFoundError(f"Blender binary not found at {self.blender_bin}")

    def render(self, run_info_path, position, forward, up):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            out_img = Path(td) / "frame.png"
            out_json = Path(td) / "frame.json"
            cmd = [
                self.blender_bin, "-b", "-P", self.render_script, "--",
                "--run_info_path", str(run_info_path),
                "--output_image", str(out_img), "--output_json", str(out_json),
                "--position", *map(str, np.asarray(position).tolist()),
                "--forward", *map(str, np.asarray(forward).tolist()),
                "--up", *map(str, np.asarray(up).tolist()),
            ]
            proc = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True,
                                  text=True, timeout=self.timeout_s)
            if proc.returncode != 0 or not out_img.exists():
                raise RuntimeError(f"Blender render failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
            return Image.open(out_img).convert("RGB")


def pose_proxy_distance(position: np.ndarray, object_position: np.ndarray,
                        target: dict[str, float], target_keys: Sequence[str]) -> Optional[float]:
    """Analytic az/el shot-profile proxy at `position` vs `target` (normalized L2).

    Same metric the trainable-policy evals report, so the rendered rollout and the
    single-step evals share a yardstick. Returns None if the target lacks az/el keys.
    """
    from src.policy.common.reward import score_distance

    needed = {"cam_to_obj_azimuth_deg", "cam_to_obj_elevation_deg"}
    if not needed.issubset(target_keys):
        return None
    vec = np.asarray(object_position, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    az = float(np.degrees(np.arctan2(vec[1], vec[0])))
    el = float(np.degrees(np.arctan2(vec[2], np.linalg.norm(vec[:2]))))
    pose_keys = [k for k in target_keys if k in needed]
    achieved = np.array([az if k == "cam_to_obj_azimuth_deg" else el for k in pose_keys], dtype=np.float32)
    tgt = np.array([target[k] for k in pose_keys], dtype=np.float32)
    return float(score_distance(achieved, tgt, pose_keys))


class BlenderRolloutEnv:
    """Gym-like camera-control env over a single scene.

    `reset(position, forward, up)` sets the start pose (optionally rendering it);
    `step(action_5d)` applies the 5D action, renders, and returns the new observation.
    The 5D action is in metres/radians (raw, un-normalized).
    """

    def __init__(self, run_info_path: str, renderer: Renderer, *,
                 object_position: Optional[Sequence[float]] = None) -> None:
        self.run_info_path = str(run_info_path)
        self.renderer = renderer
        self.object_position = (np.asarray(object_position, dtype=np.float32)
                                if object_position is not None else None)
        self.position: Optional[np.ndarray] = None
        self.forward: Optional[np.ndarray] = None
        self.up: Optional[np.ndarray] = None
        self.t = 0

    def reset(self, position, forward, up, *, render: bool = True) -> dict:
        self.position = np.asarray(position, dtype=np.float32)
        self.forward = np.asarray(forward, dtype=np.float32)
        self.up = np.asarray(up, dtype=np.float32)
        self.t = 0
        image = self.renderer.render(self.run_info_path, self.position, self.forward, self.up) if render else None
        return self._obs(image)

    def step(self, action_5d: np.ndarray, *, render: bool = True) -> tuple[dict, dict]:
        if self.position is None:
            raise RuntimeError("call reset() before step()")
        self.position, self.forward, self.up = apply_action_5d(
            self.position, self.forward, self.up, np.asarray(action_5d, dtype=np.float32))
        self.t += 1
        image = self.renderer.render(self.run_info_path, self.position, self.forward, self.up) if render else None
        return self._obs(image), {"t": self.t}

    def pose_proxy_distance(self, target: dict[str, float], target_keys: Sequence[str]) -> Optional[float]:
        if self.object_position is None or self.position is None:
            return None
        return pose_proxy_distance(self.position, self.object_position, target, target_keys)

    def _obs(self, image) -> dict:
        return {"image": image,
                "pose": {"position": self.position.copy(), "forward": self.forward.copy(), "up": self.up.copy()},
                "t": self.t}

    def close(self) -> None:
        self.renderer.close()


__all__ = ["Renderer", "MockRenderer", "SubprocessBlenderRenderer", "BlenderRolloutEnv", "pose_proxy_distance"]
