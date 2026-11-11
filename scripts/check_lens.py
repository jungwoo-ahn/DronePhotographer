"""See what the policy will see, and measure the lens.

    python scripts/check_lens.py --out lens_check              # saves wide.jpg and view.jpg
    python scripts/check_lens.py --measure                     # THE CAMERA WILL TILT

--measure tilts the gimbal a few degrees at a time and measures how far the picture moves, which
needs no chart: only something with detail in the middle of the view, and a still aircraft. Pass
the answer to the pilot as --lens-hfov. It puts the gimbal back where it found it.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.actions import TILT_MIN_DEG
from src.drone_control.camera import CameraStream
from src.drone_control.gimbal import DEFAULT_HOST, SiyiGimbal
from src.drone_control.lens import (
    A2_MINI_HFOV_DEG,
    TRAINING_HFOV_DEG,
    VIEW_SIZE,
    LensView,
    hfov_from_tilt,
    image_shift,
)

MIN_CONFIDENCE = 0.1


def wait_frame(cam, after: float, timeout_s: float = 10.0):
    """The first frame received after the monotonic time `after`."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        f = cam.latest()
        if f is not None and f.received_at > after:
            return f.image
        time.sleep(0.02)
    raise SystemExit(f"no frame from {cam.url} for {timeout_s:.0f} s")


def measure(cam, g, step_deg, steps, settle_s):
    start = g.steady_tilt()
    if start is None:
        raise SystemExit("the gimbal is not answering steadily, so nothing was moved")
    if start - steps * step_deg < TILT_MIN_DEG + 10.0:
        raise SystemExit(f"starting at {start:+.1f}° would take it near its limit, so nothing was moved")
    answers = []
    width = wait_frame(cam, time.monotonic()).shape[1]
    try:
        before, at = wait_frame(cam, time.monotonic()), start
        for n in range(1, steps + 1):
            target = start - n * step_deg
            g.set_tilt(target)
            time.sleep(settle_s)
            now = g.steady_tilt()
            after = wait_frame(cam, time.monotonic())
            if now is None or abs(now - target) > 2.0:
                raise SystemExit(f"asked for {target:+.1f}° and it reports {now}; stopping")
            dx, dy, confidence = image_shift(before, after)
            moved = now - at
            usable = confidence >= MIN_CONFIDENCE and abs(dy) > 1.0 and abs(moved) > 1.0
            hfov = hfov_from_tilt(dy, moved, width) if usable else None
            print(f"tilted {moved:+.1f}°: picture moved dx {dx:+.1f} dy {dy:+.1f} px, "
                  f"confidence {confidence:.2f}"
                  + (f" -> {hfov:.0f}°" if usable else " -> not usable"))
            if usable:
                answers.append(hfov)
            before, at = after, now
    finally:
        g.set_tilt(start)
    if len(answers) < 2:
        raise SystemExit("too few usable steps: point it at something with more detail")
    print(f"--lens-hfov {statistics.median(answers):.0f}   "
          f"(steps ranged {min(answers):.0f} to {max(answers):.0f}, the pilot's default is {A2_MINI_HFOV_DEG:.0f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera-url", default=f"rtsp://{DEFAULT_HOST}:8554/main.264")
    p.add_argument("--lens-hfov", type=float, default=A2_MINI_HFOV_DEG)
    p.add_argument("--out", default="lens_check")
    p.add_argument("--measure", action="store_true", help="tilts the camera")
    p.add_argument("--step", type=float, default=4.0, help="degrees per step, downward")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--settle", type=float, default=2.0)
    a = p.parse_args()

    cam = CameraStream(a.camera_url, max_age_s=2.0).start()
    try:
        image = wait_frame(cam, time.monotonic(), timeout_s=20.0)
        print(f"stream: {image.shape[1]}x{image.shape[0]}")
        if a.measure:
            g = SiyiGimbal()
            measure(cam, g, a.step, a.steps, a.settle)
            g.close()
            return
        import cv2
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "wide.jpg"), image)
        view = LensView(a.lens_hfov, size=VIEW_SIZE, view_hfov_deg=TRAINING_HFOV_DEG)
        cv2.imwrite(str(out / "view.jpg"), view(image))
        print(f"saved {out / 'wide.jpg'} and {out / 'view.jpg'}")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
