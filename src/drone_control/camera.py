"""Video ingest. A SECOND, independent link -- not MAVLink.

The camera streams over its own wireless router (RTSP/UDP), so the laptop now has
three failure domains that can die independently:

    MAVLink (PX4)  --  telemetry and setpoints
    video   (WiFi) --  this module
    HTTP    (SSH)  --  the policy server

Each needs its own freshness check, and none of them can detect the others' failure.
A dead video feed is the nastiest of the three, because it is invisible: frames stop
arriving, but the last decoded image is still sitting in memory and looks perfectly
valid. Acting on it means flying by a photograph of where the drone used to be --
at 3 m/s a 2 s-stale frame is 6 m of lie. Hence `Frame.age_s` and `max_age_s`, and
`ActionSequenceSource` refusing to ask the server when the picture is stale.

**Latest-frame-only.** OpenCV queues frames internally, so reading sequentially in a
~2 Hz loop hands you an ever-growing backlog: the stream plays in slow motion and
latency climbs without bound. A reader thread therefore runs flat out and keeps only
the newest frame, deliberately throwing away everything in between. For a photography
policy the current view is the only one that matters.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

#: A frame older than this is not worth acting on.
DEFAULT_MAX_AGE_S = 0.7

#: After connecting, the decoder delivers flat grey until its first keyframe: up to 1.2 s on the
#: A2 mini. Those frames look valid and fresh, so they are dropped here, with some margin.
DEFAULT_SETTLE_S = 3.0


@dataclass
class Frame:
    image: object                       # numpy array (BGR), as OpenCV delivers it
    seq: int = 0
    received_at: float = field(default_factory=time.monotonic)

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.received_at

    def is_fresh(self, max_age_s: float = DEFAULT_MAX_AGE_S, now: float | None = None) -> bool:
        return self.age_s(now) < max_age_s


def gstreamer_rtsp(url: str, latency_ms: int = 50) -> str:
    """A low-latency GStreamer pipeline, if OpenCV was built with GStreamer.

    `drop=true max-buffers=1` is the same latest-frame-only idea pushed down into the
    pipeline, which is cheaper than doing it in Python.
    """
    return (f"rtspsrc location={url} latency={latency_ms} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink drop=true max-buffers=1 sync=false")


class CameraStream:
    """Background reader that always holds the most recent frame.

    `open_capture` is injectable so the tests (and a dry run on a machine with no
    camera and no OpenCV) can drive it without any video at all.
    """

    def __init__(self, url: str, *, max_age_s: float = DEFAULT_MAX_AGE_S,
                 open_capture=None, reconnect_delay_s: float = 1.0, clock=time.monotonic,
                 settle_s: float = DEFAULT_SETTLE_S):
        self.url = url
        self.max_age_s = float(max_age_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.settle_s = float(settle_s)
        self._open = open_capture or self._default_open
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._cap = None
        self.frames_read = 0
        self.reconnects = 0

    def _default_open(self, url):  # pragma: no cover - needs OpenCV + a live camera
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "opencv-python is needed to read the camera stream:\n"
                "    pip install opencv-python\n"
                "Only the laptop needs it; the policy server does not."
            ) from e
        cap = cv2.VideoCapture(gstreamer_rtsp(url), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url)             # plain RTSP/UDP via FFMPEG
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honoured by some backends only
            except Exception:
                pass
        return cap

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "CameraStream":
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        # Release here too: the reader may be blocked in `read`, and a recording capture has a
        # file to finish properly.
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        cap = None
        while not self._stop.is_set():
            try:
                if cap is None:
                    cap = self._cap = self._open(self.url)
                    if cap is None or not cap.isOpened():
                        cap = None
                        self.reconnects += 1
                        self._stop.wait(self.reconnect_delay_s)
                        continue
                    settled_at = self._clock() + self.settle_s
                ok, img = cap.read()
                if not ok:
                    # Stream dropped. Release and retry; the old frame keeps ageing
                    # and will fail its freshness check, which is what stops the drone.
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    self.reconnects += 1
                    self._stop.wait(self.reconnect_delay_s)
                    continue
                if self._clock() < settled_at:
                    continue
                self._seq += 1
                self.frames_read += 1
                with self._lock:
                    self._frame = Frame(image=img, seq=self._seq, received_at=self._clock())
            except Exception:
                cap = None
                self.reconnects += 1
                self._stop.wait(self.reconnect_delay_s)

    # -- access ---------------------------------------------------------------
    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def fresh_frame(self, now: float | None = None) -> Frame | None:
        """The newest frame, or `None` if the feed has gone stale or never started."""
        f = self.latest()
        if f is None:
            return None
        return f if f.is_fresh(self.max_age_s, now) else None

    @property
    def stats(self) -> dict:
        f = self.latest()
        return {"frames_read": self.frames_read, "reconnects": self.reconnects,
                "last_seq": f.seq if f else None}


def encode_jpeg(image, quality: int = 80) -> bytes | None:
    """Encode on demand, not per frame: the reader runs at video rate, the policy at ~2 Hz."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
