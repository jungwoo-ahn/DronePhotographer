"""Rollout renderer for 3DGS placements: a client for `scripts/gs_render_server.py`.

Same interface as the Blender renderers in `src.common.blender_env` -- `render(run_info_path,
position, forward, up) -> PIL.Image` -- so `BlenderRolloutEnv` and the eval loop do not change.
For a 3DGS episode the "run_info" is the placement spec the episode builder wrote
(`runs/bench_gs/_specs/<placement>.json`), and the pose is in the placement's local metric frame.

The server runs in `.venv-gs` (torch cu128 + gsplat), which cannot share a process with the
policy's venv, and gets an environment of its own: the policy job's LD_LIBRARY_PATH points at
another torch's CUDA libraries, which is exactly what a second torch must not load.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from src.common.blender_env import Renderer

V12 = Path(__file__).resolve().parents[2]
SHARED = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")


def _server_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    nv = sorted((V12 / ".venv-gs" / "lib").glob("python3*/site-packages/nvidia/*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join([str(SHARED / "blender/syslibs/lib")] + [str(p) for p in nv])
    env.pop("PYTHONPATH", None)
    # A pod's CPU quota is not a cpuset: unpinned thread pools size themselves to all host cores
    # and get throttled (see gs_render_server.cpu_budget, which sizes the subject's Blender).
    n = env.get("SLURM_CPUS_ON_NODE") or "8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, n)
    return env


class GaussianCompositeRenderer(Renderer):
    """One long-lived server for a whole run: the splat stays on the GPU across a scene's
    episodes, and the subject's Blender restarts only when the placement changes."""

    def __init__(self, ready_timeout_s: float = 900.0, render_timeout_s: float = 900.0) -> None:
        self.comm = Path(tempfile.mkdtemp(prefix="gsrender_"))
        self.render_timeout_s = render_timeout_s
        # stderr to a file: an undrained pipe fills at 64 KB and blocks the server mid-run.
        self.log = self.comm / "server.log"
        with open(self.log, "w") as log:
            self.proc = subprocess.Popen(
                [str(V12 / ".venv-gs/bin/python"), str(V12 / "scripts/gs_render_server.py"),
                 "--comm", str(self.comm)],
                cwd=str(V12), env=_server_env(), stdout=subprocess.DEVNULL, stderr=log)
        t0 = time.time()
        while not (self.comm / "ready").exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"3DGS render server died on startup:\n{self._err()}")
            if time.time() - t0 > ready_timeout_s:
                raise TimeoutError("3DGS render server did not become ready")
            time.sleep(0.1)

    def _err(self) -> str:
        try:
            return self.log.read_text(errors="replace")[-2000:]
        except OSError:
            return ""

    def render(self, run_info_path, position, forward, up):
        from PIL import Image

        out = self.comm / f"frame_{time.time_ns()}.png"
        resp = self.comm / "response.json"
        resp.unlink(missing_ok=True)
        tmp = self.comm / "request.tmp"
        tmp.write_text(json.dumps({"spec": str(run_info_path), "output_image": str(out),
                                   "position": [float(v) for v in position],
                                   "forward": [float(v) for v in forward],
                                   "up": [float(v) for v in up]}))
        tmp.replace(self.comm / "request.json")
        t0 = time.time()
        while not resp.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"3DGS render server died:\n{self._err()}")
            if time.time() - t0 > self.render_timeout_s:
                raise TimeoutError("3DGS render timed out")
            time.sleep(0.02)
        r = json.loads(resp.read_text())
        resp.unlink(missing_ok=True)
        if not r.get("ok"):
            raise RuntimeError(f"3DGS render failed: {r.get('error')}")
        img = Image.open(r["image"]).convert("RGB")
        Path(r["image"]).unlink(missing_ok=True)
        return img

    def close(self) -> None:
        try:
            (self.comm / "stop").write_text("1")
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        shutil.rmtree(self.comm, ignore_errors=True)


__all__ = ["GaussianCompositeRenderer"]
