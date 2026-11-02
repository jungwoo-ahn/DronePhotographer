"""Render a camera pose in a 3DGS placement: splat background + Blender subject, depth-composited.

The closed-loop eval asks for a picture at every chunk end, from wherever the policy flew; the
3DGS session's placement pipeline (`data/gs_scenes/_tools/composite`) made one still per
placement. This renders the same composite for ANY pose, with the same pieces that made those
stills, so a rollout frame, the goal image and the start image all come out of one renderer:
  * background  the trained splat through gsplat (`gscommon.Splat`, unchanged), RGB + depth
  * subject     `scripts/gs_subject_blender.py serve`, one Blender per placement, kept alive
  * composite   a subject pixel survives where it is nearer than the splat surface (15 cm slack,
                so feet on uneven splat ground do not flicker) or where the splat is empty --
                the rule of the pipeline's own `compose.py`

Poses arrive in the placement's local metric frame L (feet at the origin, z up, +x toward the
preview camera) -- the frame the policy acts in. L is rebuilt from the job's azimuth exactly as
`prepare_jobs.py` built it, which `--check` verifies against the stored preview composites.

As a server (spawned by `src/common/gs_render.GaussianCompositeRenderer`):
    .venv-gs/bin/python scripts/gs_render_server.py --comm DIR
As a library: `Compositor().render(spec_path, position, forward, up)`.
venv: .venv-gs (torch cu128 + gsplat).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, f"{SHARED}/data/gs_scenes/_tools/composite")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

W, H = 1024, 768
HFOV = math.degrees(2 * math.atan(6.4 / 24.0))     # 24 mm on 12.8 mm: the v7 lens, as Blender sees it
SLACK_M = 0.15
# A splat has no sky: where nothing was reconstructed, gsplat leaves the background black, and a
# low camera looking up at the subject shows a black void over its head. Fill it with the sky
# colour prepare_jobs.py already uses for the environment map that lights the subject, so the
# backdrop and the lighting agree. All five scenes are outdoors.
SKY = np.array([205, 220, 240], np.float32) / 255.0


def local_frame(up, azimuth_deg: float) -> np.ndarray:
    """Columns = the L axes in splat coordinates, rebuilt the way prepare_jobs.py chose them."""
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)

    def hor(v):
        return v - (v @ up) * up
    e1 = hor(np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    a = math.radians(azimuth_deg)
    x = math.cos(a) * e1 + math.sin(a) * e2
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up], 1)


def camera_axes(forward, up):
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)                       # forward, right, true up


def cpu_budget() -> int:
    """CPUs this job may use. A worker pod's quota is a cgroup CPU limit, not a cpuset, so every
    process sees all 144 host cores, sizes its thread pools to them and is then throttled to the
    8 it was given: Blender ran ~870 threads in an 8-core pod and took ~16 s a frame instead of
    0.4 s. Everything spawned here gets this number instead."""
    return int(os.environ.get("SLURM_CPUS_ON_NODE") or 8)


class SubjectServer:
    """One Blender holding one placement's subject scene."""

    def __init__(self, spec_path: str):
        self.comm = Path(tempfile.mkdtemp(prefix="gssubj_"))
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{SHARED}/blender/syslibs/lib:" + env.get("LD_LIBRARY_PATH", "")
        # stderr to a file, not a pipe: nothing reads a pipe until the process dies, and a full
        # 64 KB pipe blocks the writer -- a Blender that warns enough would hang mid-run.
        self.log = self.comm / "blender.log"
        with open(self.log, "w") as log:
            self.proc = subprocess.Popen(
                [f"{SHARED}/blender/blender", "-b", "-t", str(cpu_budget()), "-P",
                 f"{V12}/scripts/gs_subject_blender.py", "--", "serve", spec_path, str(self.comm)],
                cwd=SHARED, env=env, stdout=subprocess.DEVNULL, stderr=log)
        t0 = time.time()
        while not (self.comm / "ready").exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"subject server died on startup:\n{self._err()}")
            if time.time() - t0 > 900:
                raise TimeoutError("subject server did not become ready")
            time.sleep(0.1)

    def render(self, location, rotation) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        import OpenEXR

        fg = self.comm / f"fg_{time.time_ns()}.png"
        resp = self.comm / "response.json"
        resp.unlink(missing_ok=True)
        tmp = self.comm / "request.tmp"
        tmp.write_text(json.dumps({"location": list(map(float, location)),
                                   "rotation": [list(map(float, r)) for r in rotation], "fg": str(fg)}))
        tmp.replace(self.comm / "request.json")
        t0 = time.time()
        while not resp.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"subject server died:\n{self._err()}")
            if time.time() - t0 > 900:
                raise TimeoutError("subject render timed out")
            time.sleep(0.01)
        r = json.loads(resp.read_text())
        resp.unlink(missing_ok=True)
        if not r.get("ok"):
            raise RuntimeError(f"subject render failed: {r.get('error')}")
        rgba = cv2.imread(r["fg"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        rgba = rgba[..., [2, 1, 0, 3]]                                    # BGRA -> RGBA
        z = next(iter(OpenEXR.File(r["z"]).channels().values())).pixels
        z = z[..., 0] if z.ndim == 3 else z
        for p in (r["fg"], r["z"]):
            Path(p).unlink(missing_ok=True)
        return rgba, np.asarray(z, np.float32)

    def _err(self) -> str:
        try:
            return self.log.read_text(errors="replace")[-2000:]
        except OSError:
            return ""

    def close(self):
        try:
            (self.comm / "stop").write_text("1")
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class Compositor:
    """Splat + subject composite for any pose in any placement spec; caches the current scene."""

    def __init__(self):
        self.ply = None
        self.splat = None
        self.spec_path = None
        self.subject = None

    def _load(self, spec_path: str) -> None:
        from gscommon import Splat

        spec = json.loads(Path(spec_path).read_text())
        if spec["ply"] != self.ply:
            self.splat = None
            import torch
            torch.cuda.empty_cache()
            self.splat, self.ply = Splat(spec["ply"]), spec["ply"]
        if self.subject is not None:
            self.subject.close()
        self.subject = SubjectServer(spec_path)
        self.spec, self.spec_path = spec, spec_path
        self.mpu = float(spec["mpu"])
        self.S = np.asarray(spec["spot_splat"], float)
        self.Rl = local_frame(spec["up"], spec["azimuth_deg"])

    def render(self, spec_path: str, position, forward, up) -> np.ndarray:
        from gscommon import K_hfov

        if spec_path != self.spec_path:
            self._load(spec_path)
        f, r, u = camera_axes(forward, up)
        p = np.asarray(position, float)
        # Background: the pose in splat coordinates, OpenCV axes (x right, y down, z forward).
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = self.Rl @ r, -(self.Rl @ u), self.Rl @ f
        c2w[:3, 3] = self.S + self.Rl @ p / self.mpu
        rgb, depth, alpha = self.splat.render(c2w, K_hfov(HFOV, W, H), W, H, near=0.05 / self.mpu)
        if os.environ.get("GS_BACKGROUND", "splat") == "gray":
            # Diagnostic: the subject (and its shadow) over flat mid-gray instead of the splat --
            # separates "the photographic background" from "the subject" as the cause of a failure.
            rgb = np.full_like(rgb, 128)
            depth = np.full_like(depth, np.inf)
            alpha = np.ones_like(alpha)
        # Subject: the same pose in L, Blender camera axes (right, up, -forward).
        fg, fz = self.subject.render(p, np.stack([r, u, -f], 1))
        a = fg[..., 3] * ((fz < depth * self.mpu + SLACK_M) | (alpha < 0.5))
        bg = rgb.astype(np.float32) / 255.0 + (1 - alpha[..., None]) * SKY      # rgb is over black
        out = bg.clip(0, 1) * (1 - a[..., None]) + fg[..., :3] * a[..., None]
        return (out.clip(0, 1) * 255).astype(np.uint8)

    def close(self):
        if self.subject is not None:
            self.subject.close()
            self.subject = None


def serve(comm: Path) -> None:
    from PIL import Image

    comp = Compositor()
    (comm / "ready").write_text("1")
    req, resp, stop = comm / "request.json", comm / "response.json", comm / "stop"
    parent = os.getppid()                      # the eval; when it dies without `stop`, so do we
    try:
        while not stop.exists() and os.getppid() == parent:
            if not req.exists():
                time.sleep(0.02)
                continue
            try:
                r = json.loads(req.read_text())
            except Exception:  # noqa: BLE001
                time.sleep(0.02)
                continue
            req.unlink(missing_ok=True)
            try:
                img = comp.render(r["spec"], r["position"], r["forward"], r["up"])
                Image.fromarray(img).save(r["output_image"])
                payload = {"ok": True, "image": r["output_image"]}
            except Exception as e:  # noqa: BLE001
                payload = {"ok": False, "error": repr(e)}
            tmp = resp.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(resp)
    finally:
        comp.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm", required=True)
    serve(Path(ap.parse_args().comm))
