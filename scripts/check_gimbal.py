"""Try the gimbal on the bench: read the angle, tilt, read it back, and centre.

    python scripts/check_gimbal.py --tilt -30

The camera will move, and nothing else does. This needs no drone link and no server.

It also answers the one question the manual does not, with --watch: tip the aircraft by hand,
nose down, with the propellers off, and see what the angle does. If it stays put, the gimbal
reports against the horizon, which is what flying needs. If it follows, it reports against
the airframe.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.gimbal import DEFAULT_HOST, DEFAULT_PORT, SiyiGimbal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--tilt", type=float, default=-30.0, help="degrees, up positive")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to let it get there")
    p.add_argument("--watch", type=float, default=0.0, help="only print the angle, for this long")
    p.add_argument("--main-stream", choices=["720", "1080"], default=None, dest="main_stream",
                   help="set the video stream's size, and move nothing")
    p.add_argument("--kbps", type=int, default=4096, help="the video stream's bitrate")
    a = p.parse_args()

    g = SiyiGimbal(a.host, a.port)
    if a.main_stream:
        width, height = (1920, 1080) if a.main_stream == "1080" else (1280, 720)
        print(f"stream now: {g.main_stream()}")
        accepted = g.set_main_stream(width, height, a.kbps)
        print(f"asking for {width}x{height} at {a.kbps} kbps, accepted: {accepted}")
        if accepted:
            # The stream serves no picture at the new size until the camera restarts.
            print(f"restarting the camera (not the gimbal): {g.reboot_camera()}. "
                  "The picture is back in about a minute.")
        time.sleep(a.settle)
        print(f"stream after: {g.main_stream()}. Check the picture really changed size:")
        print("    python scripts/check_lens.py --out lens_check")
        g.close()
        return

    if a.watch:
        end = time.monotonic() + a.watch
        while time.monotonic() < end:
            print(f"{g.tilt()}", flush=True)
            time.sleep(0.5)
        return

    start = g.steady_tilt()
    if start is None:
        raise SystemExit(f"no steady answer from {a.host}:{a.port}, so nothing was moved. "
                         "Can this machine ping it?")
    print(f"tilt now: {start:+.1f}°")
    print(f"asking for {a.tilt:+.1f}°, answered {g.set_tilt(a.tilt)}")
    time.sleep(a.settle)
    print(f"tilt after {a.settle:.0f} s: {g.tilt()}. Is the camera pointing that way, down negative?")
    print(f"centring: {g.center()}")
    time.sleep(a.settle)
    print(f"tilt after centring: {g.tilt()}")
    g.close()


if __name__ == "__main__":
    main()
