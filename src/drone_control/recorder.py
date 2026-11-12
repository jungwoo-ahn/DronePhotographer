"""Recording the flight: the camera's own stream, saved untouched, for post-processing.

ffmpeg copies the packets the camera sends (H.265, 1920x1080 at about 4 Mbps) into a Matroska
file, so the video is exactly what the camera encoded, nothing is re-encoded, and it costs
almost no CPU. Matroska, because a file cut short by a crash or a pulled battery still plays.
For an editor that wants MP4:  ffmpeg -i video.mkv -c copy -tag:v hvc1 video.mp4

**One connection to the camera, not two.** The A2 mini's video server hung after about 25 s
with two clients (the pilot's and a recorder's), and froze both. So when recording, the same
ffmpeg that writes the file also decodes the frames the pilot sees: `RecordingCapture` stands
in for OpenCV's capture inside `CameraStream`. Each reconnection starts a new part,
`video.mkv`, `video-2.mkv` and so on.

ffmpeg reports how far into the file it is; each report goes into the flight log, stamped
with the laptop's clock, so any moment of the video can be matched to what was flying.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPORT_EVERY_S = 5.0        # how often a video-time report goes into the log
STALL_S = 5.0               # no frame for this long: the stream is dead, so reconnect

_JOB = None


def tie_to_this_process(proc) -> bool:
    """On Windows, put `proc` in a job that is killed when this process ends, however it ends.

    Otherwise a pilot killed from Task Manager, or crashed, would leave ffmpeg recording until
    the camera died. A normal stop still asks ffmpeg to finish its file first.
    """
    global _JOB
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if _JOB is None:
        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.Basic.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                                      ctypes.sizeof(info)):
            return False
        _JOB = job                              # held until this process ends, on purpose
    handle = getattr(proc, "_handle", None)
    return bool(handle) and bool(k32.AssignProcessToJobObject(_JOB, int(handle)))


def find_ffmpeg() -> str | None:
    """ffmpeg on the PATH, or the one `pip install imageio-ffmpeg` brings."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def capture_command(ffmpeg: str, url: str, record_to: Path, size: tuple[int, int],
                    transport: str = "tcp") -> list[str]:
    """One input, two outputs: the stream copied to `record_to`, and BGR frames on stdout.

    RTSP over TCP by default: outdoors over UDP, lost packets broke frames every 10-20 s
    ("Error constructing the frame RPS"), and the picture froze until the next keyframe.
    TCP sends a lost packet again instead.
    """
    w, h = size
    rtsp = ["-rtsp_transport", transport] if url.startswith("rtsp://") else []
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-progress", "pipe:2", "-nostats",
            *rtsp, "-fflags", "+genpts", "-i", url,   # not +nobuffer: nothing decodes with it
            "-map", "0:v:0", "-c", "copy", "-f", "matroska", "-y", str(record_to),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "pipe:1"]


class RecordingCapture:
    """What `CameraStream` reads from while recording: `isOpened`, `read`, `release`, like
    OpenCV's capture, from one ffmpeg that also writes the stream to a file."""

    def __init__(self, cmd: list[str], size: tuple[int, int], *, name: str, path: Path | None = None,
                 log=None, popen=subprocess.Popen, clock=time.time, stall_s: float = STALL_S):
        self.size = size
        self.name = name
        self.path = path
        self.log = log
        self._clock = clock
        self._frame_bytes = size[0] * size[1] * 3
        self.seconds = 0.0
        self.error: str | None = None
        self._last_frame = time.monotonic()
        self._done = threading.Event()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, creationflags=flags, bufsize=0)
        tie_to_this_process(self.proc)
        self._event("video_start", file=name)
        threading.Thread(target=self._progress, name="ffmpeg-progress", daemon=True).start()
        threading.Thread(target=self._watch, args=(stall_s,), name="ffmpeg-watch",
                         daemon=True).start()

    def _event(self, kind, **data):
        if self.log is not None:
            self.log.event(kind, **data)

    def isOpened(self) -> bool:                       # noqa: N802 - OpenCV's name
        return self.proc.poll() is None

    def read(self):
        """The next frame as a numpy array, or `(False, None)` once the stream has ended."""
        import numpy as np

        buf = bytearray(self._frame_bytes)
        view, got = memoryview(buf), 0
        while got < self._frame_bytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                return False, None
            got += n
        self._last_frame = time.monotonic()
        return True, np.frombuffer(buf, np.uint8).reshape(self.size[1], self.size[0], 3)

    def _drain(self):
        try:
            while self.proc.stdout.read(1 << 20):
                pass
        except (OSError, ValueError):
            pass

    def _progress(self):
        """ffmpeg's stderr: progress reports (how long the file is) and any errors."""
        reported = None
        errors = []
        for raw in self.proc.stderr:
            key, sep, value = raw.decode(errors="replace").strip().partition("=")
            if sep and key == "out_time_us" and value.lstrip("-").isdigit():
                self.seconds = max(0.0, int(value) / 1e6)
            elif sep and key == "progress":
                now = self._clock()
                if reported is None or now - reported >= REPORT_EVERY_S:
                    reported = now
                    self._event("video_time", file=self.name, seconds=round(self.seconds, 3))
            elif not sep or " " in key:
                errors.append(raw.decode(errors="replace").strip())
        code = self.proc.wait()
        if errors:
            self.error = errors[-1][:200]
        # A connection that never got a picture leaves an empty file; one per failed retry
        # once left 140 of them. Remove it, and the next attempt takes its name.
        removed = False
        if self.seconds < 0.5 and self.path is not None:
            try:
                if self.path.exists() and self.path.stat().st_size < (1 << 20):
                    self.path.unlink()
                    removed = True
            except OSError:
                pass
        self._event("video_end", file=self.name, seconds=round(self.seconds, 3), code=code,
                    errors=errors[-5:], removed=removed)
        self._done.set()

    def _watch(self, stall_s: float):
        """A camera that stops sending leaves ffmpeg waiting forever: end it, so the pilot
        sees the stream drop and reconnects into a new part."""
        while not self._done.wait(1.0):
            if time.monotonic() - self._last_frame > stall_s and self.proc.poll() is None:
                self.error = f"no video for {stall_s:.0f} s"
                self.proc.kill()

    def release(self, timeout_s: float = 5.0):
        """Ask ffmpeg to finish the file properly (it reads `q`), then make sure it has.

        Frames are drained meanwhile: once nobody reads them, ffmpeg blocks writing the next
        one and never gets round to reading the `q`, so the file is never finished.
        """
        if self.proc.poll() is None:
            threading.Thread(target=self._drain, name="ffmpeg-drain", daemon=True).start()
            try:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._done.wait(timeout_s)


class RecordingOpener:
    """Opens the camera for `CameraStream`, recording each connection as the next part."""

    def __init__(self, out_dir, size: tuple[int, int] = (1920, 1080), *, ffmpeg: str | None = None,
                 log=None, popen=subprocess.Popen, stall_s: float = STALL_S,
                 transport: str = "tcp"):
        self.out_dir = Path(out_dir)
        self.size = size
        self.transport = transport
        self.ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        self.log = log
        self._popen = popen
        self._stall_s = stall_s
        self.parts: list[str] = []
        self.current: RecordingCapture | None = None
        self.error: str | None = None if self.ffmpeg else "no ffmpeg; pip install imageio-ffmpeg"

    def __call__(self, url: str):
        if self.ffmpeg is None:
            raise RuntimeError(self.error)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while (self.out_dir / part_name(n)).exists():
            n += 1                      # empty parts are removed, so their names come back
        name = part_name(n)
        if name not in self.parts:
            self.parts.append(name)
        path = self.out_dir / name
        self.current = RecordingCapture(capture_command(self.ffmpeg, url, path, self.size,
                                                        self.transport),
                                        self.size, name=name, path=path, log=self.log,
                                        popen=self._popen, stall_s=self._stall_s)
        return self.current

    def status(self) -> dict:
        cap = self.current
        live = cap is not None and cap.isOpened() and cap.seconds > 0
        kept = [p for p in self.parts if (self.out_dir / p).exists()]
        return {"recording": live, "file": cap.name if cap else None,
                "parts": len(kept), "seconds": cap.seconds if cap else 0.0,
                "error": None if live else (cap.error if cap else self.error),
                "dir": str(self.out_dir.resolve())}


def part_name(n: int) -> str:
    return "video.mkv" if n == 1 else f"video-{n}.mkv"


def _part_number(path: Path) -> int:
    stem = path.stem
    return 1 if stem == "video" else int(stem.split("-")[-1])


def make_mp4(out_dir, ffmpeg: str | None = None, name: str = "flight.mp4") -> Path | None:
    """Join a flight's recorded parts into one MP4 that players can seek in.

    A recording cut short (a dropped link, a stopped process) has no index, so VLC plays it
    but cannot jump around. Rewriting it as MP4 builds the index; the picture is copied, not
    re-encoded, and the parts are kept. `None` if there is nothing to join.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    out_dir = Path(out_dir)
    parts = sorted((p for p in out_dir.glob("video*.mkv") if p.stat().st_size > 4096),
                   key=_part_number)
    if ffmpeg is None or not parts:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(parts[0])], capture_output=True,
                           text=True, creationflags=flags).stderr
    tag = ["-tag:v", "hvc1"] if "Video: hevc" in probe else []   # what Apple and Windows expect
    pieces = []
    for k, part in enumerate(parts, 1):
        piece = out_dir / f"_piece{k}.mp4"
        done = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-fflags",
                               "+genpts", "-i", str(part), "-map", "0:v:0", "-c", "copy", *tag,
                               str(piece)], capture_output=True, creationflags=flags)
        if done.returncode == 0 and piece.exists() and piece.stat().st_size > 0:
            pieces.append(piece)
    if not pieces:
        return None
    listing = out_dir / "_pieces.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in pieces), encoding="utf-8")
    target = out_dir / name
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe",
                    "0", "-i", str(listing), "-c", "copy", *tag, "-movflags", "+faststart",
                    str(target)], capture_output=True, creationflags=flags)
    for p in pieces + [listing]:
        p.unlink(missing_ok=True)
    return target if target.exists() and target.stat().st_size > 0 else None


__all__ = ["RecordingOpener", "RecordingCapture", "capture_command", "find_ffmpeg",
           "make_mp4", "part_name", "tie_to_this_process"]
