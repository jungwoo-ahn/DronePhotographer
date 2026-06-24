"""PersistentBlenderRenderer — keeps one Blender process alive for fast RL rollouts.

Implements the `blender_env.Renderer` interface by talking to a long-lived
`scripts/blender_render_worker.py` over stdin/stdout, so each frame costs only a
render (no per-frame scene load). Scene-bound: constructed for one run_info; if a
different run_info is requested it restarts the worker. EEVEE + low samples make
RL-scale rollouts tractable.

(The Blender-backed path is verified on the render machine; no Blender in CI. The
pure-Python env/RL logic is covered by tests with MockRenderer.)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.policy.common.blender_env import Renderer


class PersistentBlenderRenderer(Renderer):
    def __init__(self, blender_bin: str = "blender/blender",
                 worker_script: str = "scripts/blender_render_worker.py",
                 repo_root: Path | None = None, *, engine: str = "BLENDER_EEVEE_NEXT",
                 samples: int = 16) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self.blender_bin = str((self.repo_root / blender_bin).resolve())
        self.worker_script = str((self.repo_root / worker_script).resolve())
        self.engine = engine
        self.samples = samples
        self._proc: subprocess.Popen | None = None
        self._run_info: str | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="autophoto_frames_")
        self._n = 0

    def _start(self, run_info_path: str) -> None:
        self._stop()
        cmd = [self.blender_bin, "-b", "-P", self.worker_script, "--",
               "--run_info_path", str(run_info_path), "--engine", self.engine,
               "--samples", str(self.samples)]
        self._proc = subprocess.Popen(cmd, cwd=str(self.repo_root), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1)
        self._run_info = str(run_info_path)
        # wait for the {"ready": true} handshake
        for line in self._proc.stdout:
            if '"ready"' in line:
                break

    def _readline_ok(self) -> dict:
        for line in self._proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError("Blender worker closed unexpectedly")

    def render(self, run_info_path, position, forward, up):
        from PIL import Image

        if self._proc is None or self._run_info != str(run_info_path):
            self._start(str(run_info_path))
        out = Path(self._tmpdir) / f"f{self._n}.png"
        self._n += 1
        req = {"position": list(map(float, position)), "forward": list(map(float, forward)),
               "up": list(map(float, up)), "out": str(out)}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = self._readline_ok()
        if not resp.get("ok"):
            raise RuntimeError(f"render failed: {resp.get('error')}")
        return Image.open(resp["path"]).convert("RGB")

    def _stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        self._stop()


__all__ = ["PersistentBlenderRenderer"]
