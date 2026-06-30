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
import shutil
import subprocess
import tempfile
import time
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


class PersistentBlenderRenderer(Renderer):
    """Keep one Blender process alive per scene and render poses via file IPC.

    The scene load + BVH build + GPU upload + kernel compile happen once (first render
    ~30-40s); subsequent renders only move the camera, so they are ~1-2s — ~10-15x
    faster than `SubprocessBlenderRenderer` for multi-step rollouts. Same interface;
    a changed `run_info_path` (new scene) transparently restarts the server.
    """

    def __init__(
        self,
        blender_bin: str = "blender/blender",
        server_script: str = "scripts/blender_render_server.py",
        repo_root: Optional[Path] = None,
        ready_timeout_s: float = 900.0,    # first startup may compile sm_100 kernels
        render_timeout_s: float = 300.0,   # first render builds BVH; later ones are quick
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.blender_bin = str((self.repo_root / blender_bin).resolve())
        self.server_script = str((self.repo_root / server_script).resolve())
        self.ready_timeout_s = ready_timeout_s
        self.render_timeout_s = render_timeout_s
        if not Path(self.blender_bin).exists():
            raise FileNotFoundError(f"Blender binary not found at {self.blender_bin}")
        self._proc: Optional[subprocess.Popen] = None
        self._comm: Optional[Path] = None
        self._cur_run_info: Optional[str] = None

    def _server_err(self) -> str:
        return (self._proc.stderr.read()[-2000:] if self._proc and self._proc.stderr else "")

    def _start(self, run_info_path) -> None:
        self._stop()
        self._comm = Path(tempfile.mkdtemp(prefix="blendsrv_"))   # local /tmp -> fast IPC
        cmd = [self.blender_bin, "-b", "-P", self.server_script, "--",
               "--run_info_path", str(run_info_path), "--comm_dir", str(self._comm)]
        self._proc = subprocess.Popen(cmd, cwd=str(self.repo_root),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        ready = self._comm / "ready"
        t0 = time.time()
        while not ready.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(f"render server died on startup (rc={self._proc.returncode}):\n{self._server_err()}")
            if time.time() - t0 > self.ready_timeout_s:
                self._stop()
                raise TimeoutError("render server did not become ready")
            time.sleep(0.1)
        self._cur_run_info = str(run_info_path)

    def render(self, run_info_path, position, forward, up):
        from PIL import Image

        if str(run_info_path) != self._cur_run_info:
            self._start(run_info_path)
        assert self._comm is not None and self._proc is not None
        out_img = self._comm / f"frame_{time.time_ns()}.png"
        resp = self._comm / "response.json"
        resp.unlink(missing_ok=True)
        req = self._comm / "request.json"
        tmp = req.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "position": np.asarray(position).tolist(),
            "forward": np.asarray(forward).tolist(),
            "up": np.asarray(up).tolist(),
            "output_image": str(out_img),
        }))
        tmp.replace(req)   # atomic -> server never reads a partial request
        t0 = time.time()
        while not resp.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(f"render server died (rc={self._proc.returncode}):\n{self._server_err()}")
            if time.time() - t0 > self.render_timeout_s:
                raise TimeoutError("render timed out")
            time.sleep(0.02)
        r = json.loads(resp.read_text())
        resp.unlink(missing_ok=True)
        if not r.get("ok"):
            raise RuntimeError(f"server render error: {r.get('error')}")
        img = Image.open(r["image"]).convert("RGB")
        Path(r["image"]).unlink(missing_ok=True)
        return img

    def _stop(self) -> None:
        if self._proc is not None:
            try:
                if self._comm is not None:
                    (self._comm / "stop").write_text("1")
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._comm is not None:
            shutil.rmtree(self._comm, ignore_errors=True)
            self._comm = None
        self._cur_run_info = None

    def close(self) -> None:
        self._stop()


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
        self.sample = None            # set by from_validation_sample
        self._owns_run_info = False   # whether to clean up a temp run_info on close

    @classmethod
    def from_validation_sample(cls, sample, renderer: Renderer, *,
                               run_info_path: Optional[str] = None) -> "BlenderRolloutEnv":
        """Set up the env for a `ValidationSample` (issue #23 points 1-2).

        Writes the sample's (extended) run_info — `scene_scale` + object transform +
        intrinsics — so the setup-aware renderer reconstructs the exact scene/object,
        and primes `object_position` (subject center) for the pose proxy. Use
        `reset_to_start(pair_idx)` to begin at a recorded start pose.
        """
        import os
        import tempfile

        owns = run_info_path is None
        if owns:
            fd, run_info_path = tempfile.mkstemp(suffix="_run_info.json",
                                                 prefix=f"{sample.placement}_")
            os.close(fd)
        Path(run_info_path).write_text(json.dumps(sample.to_run_info(), indent=2))
        env = cls(run_info_path, renderer, object_position=sample.subject_center)
        env.sample = sample
        env._owns_run_info = owns
        return env

    def reset_to_start(self, pair_idx: int = 0, *, render: bool = True) -> dict:
        """Reset to a recorded start pose of the loaded validation sample."""
        if self.sample is None:
            raise RuntimeError("reset_to_start requires from_validation_sample()")
        pos, fwd, up = self.sample.start_pose(pair_idx)
        return self.reset(pos, fwd, up, render=render)

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
        if self._owns_run_info:
            try:
                Path(self.run_info_path).unlink()
            except OSError:
                pass


__all__ = ["Renderer", "MockRenderer", "SubprocessBlenderRenderer",
           "BlenderRolloutEnv", "pose_proxy_distance"]
