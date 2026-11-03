"""Fly PX4 SITL from the laptop. Milestone 1 uses a stub source, no policy needed.

    # terminal 1 -- PX4 SITL
    make px4_sitl gz_x500

    # terminal 2 -- this, flying a square
    python scripts/run_sitl_pilot.py --source square --duration 40

    # later, against the policy server through an SSH tunnel
    ssh -N -L 8777:localhost:8777 <user>@<server>
    python scripts/run_sitl_pilot.py --source policy --policy-url http://127.0.0.1:8777/act

Ctrl-C is an e-stop: it hovers, leaves offboard and lands, rather than dropping the
link and letting PX4's own failsafe decide.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.approval import ApprovalGate
from src.drone_control.camera import CameraStream
from src.drone_control.loop import run
from src.drone_control.offboard import OffboardDriver
from src.drone_control.policy_client import PolicyClient
from src.drone_control.safety import SafetyLimits
from src.drone_control.sources import HoverSource, SquareSource, VisionPolicySource
from src.drone_control.supervisor import Supervisor
from src.drone_control.ui import UiServer, UiState


def build_source(a):
    """Returns (source, camera-or-None). The camera is a separate link from MAVLink."""
    if a.source == "square":
        return SquareSource(speed=a.speed), None
    if a.source == "hover":
        return HoverSource(), None
    client = PolicyClient(url=a.policy_url, timeout_s=a.policy_timeout)
    if not a.camera_url:
        # Without a camera the policy has nothing to look at. Refuse rather than fly
        # a vision policy blind.
        raise SystemExit("--source policy needs --camera-url (the camera is on its own "
                         "wireless link, not the MAVLink channel)")
    cam = CameraStream(a.camera_url, max_age_s=a.max_frame_age).start()
    return VisionPolicySource(cam, client, goal={"brief": a.brief}), cam


async def main_async(a):
    driver = OffboardDriver(a.url)
    print(f"[pilot] connecting to {a.url} ...")
    await driver.connect()
    print("[pilot] waiting for a usable position estimate ...")
    await driver.wait_until_ready()
    print(f"[pilot] arming, taking off to {a.altitude} m")
    await driver.arm_and_takeoff(a.altitude)
    await asyncio.sleep(a.takeoff_settle)

    sup = Supervisor(SafetyLimits(max_horizontal_speed=a.speed,
                                  max_altitude=a.max_altitude,
                                  max_radius=a.max_radius))
    await driver.start()
    print(f"[pilot] offboard engaged; source={a.source}")
    source, cam = build_source(a)
    client = getattr(source, "client", None)
    gate = None
    if a.approve:
        # Nothing the policy proposes reaches the aircraft until someone clicks Run,
        # so this only makes sense with the panel open.
        gate = ApprovalGate(source, limits=sup.limits, hold_s=a.move_seconds)
        source = gate
    ui = uisrv = None
    if a.ui:
        ui = UiState(sup.limits)
        uisrv = UiServer(ui, sup, port=a.ui_port, gate=gate).start()
        print(f"[pilot] panel at {uisrv.url}")
    try:
        stats = await run(driver, sup, source, policy_hz=a.policy_hz, duration_s=a.duration,
                          ui=ui, camera=cam, client=client, gate=gate)
        print(f"[pilot] {stats}")
        if cam is not None:
            print(f"[pilot] camera {cam.stats}")
    except KeyboardInterrupt:
        print("\n[pilot] Ctrl-C -> e-stop")
        sup.estop()
    finally:
        print("[pilot] landing")
        await driver.stop_and_land()
        if cam is not None:
            cam.stop()
        if uisrv is not None:
            uisrv.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="udp://:14540", help="PX4 connection (SITL default)")
    p.add_argument("--source", default="square", choices=["square", "hover", "policy"])
    p.add_argument("--policy-url", default="http://127.0.0.1:8777/act", dest="policy_url")
    p.add_argument("--policy-timeout", type=float, default=0.8, dest="policy_timeout")
    p.add_argument("--policy-hz", type=float, default=2.0, dest="policy_hz")
    p.add_argument("--altitude", type=float, default=5.0)
    p.add_argument("--takeoff-settle", type=float, default=8.0, dest="takeoff_settle")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--max-altitude", type=float, default=30.0, dest="max_altitude")
    p.add_argument("--max-radius", type=float, default=50.0, dest="max_radius")
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--camera-url", default=None, dest="camera_url",
                   help="RTSP/UDP URL of the camera on its own wireless link, "
                        "e.g. rtsp://192.168.1.50:554/live")
    p.add_argument("--max-frame-age", type=float, default=0.7, dest="max_frame_age",
                   help="older than this and the policy is not asked -> hover")
    p.add_argument("--brief", default="", help="the shot to achieve (text goal)")
    p.add_argument("--approve", action="store_true",
                   help="fly move by move: the policy proposes, you click Run in the panel")
    p.add_argument("--move-seconds", type=float, default=1.5, dest="move_seconds",
                   help="how long one accepted move flies before the aircraft holds")
    p.add_argument("--ui", action="store_true", help="serve the ground-station panel on localhost")
    p.add_argument("--ui-port", type=int, default=8080, dest="ui_port")
    a = p.parse_args()
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\n[pilot] interrupted before offboard; nothing to land")


if __name__ == "__main__":
    main()
