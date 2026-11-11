"""Fly PX4 from the laptop. Milestone 1 uses a stub source, no policy needed.

    # terminal 1 -- PX4 SITL
    make px4_sitl gz_x500

    # terminal 2 -- this, flying a square
    python scripts/run_sitl_pilot.py --source square --duration 40

    # later, against the policy server through an SSH tunnel
    ssh -N -L 8777:localhost:8777 <user>@<server>
    python scripts/run_sitl_pilot.py --source policy --policy-url http://127.0.0.1:8777/act

Ctrl-C is an e-stop: it hovers, leaves offboard and lands, rather than dropping the
link and letting PX4's own failsafe decide.

Without flying at all:

    python scripts/run_sitl_pilot.py --dry-run --approve --ui --duration 600   # no hardware
    python scripts/run_sitl_pilot.py --bench --url ... --ui --duration 600     # props off

Everything that can refuse does so on the ground: nothing below arms the aircraft until
the panel is up, the camera has delivered a frame and the policy has answered.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.approval import DEFAULT_OFFER_S, ApprovalGate
from src.drone_control.camera import CameraStream
from src.drone_control.gimbal import SiyiGimbal, TiltHold
from src.drone_control.lens import A2_MINI_HFOV_DEG, TRAINING_HFOV_DEG, VIEW_SIZE, LensView
from src.drone_control.loop import run
from src.drone_control.offboard import BenchDriver, DryRunDriver, OffboardDriver
from src.drone_control.policy_client import PolicyClient
from src.drone_control.safety import SafetyLimits
from src.drone_control.shots import load_shots
from src.drone_control.sources import ActionSequenceSource, HoverSource, SquareSource
from src.drone_control.supervisor import Supervisor
from src.drone_control.tunnel import ServerTunnel, load_server
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
    if not a.goal_file:
        raise SystemExit("--source policy needs --goal-file: the shot profile to ask for")
    with open(a.goal_file, encoding="utf-8") as f:
        goal = json.load(f)
    host = urlparse(a.camera_url).hostname
    gimbal = SiyiGimbal(host) if host and (a.gimbal or a.full_hd) else None
    if gimbal is not None and a.full_hd and use_full_hd(gimbal, a.stream_kbps):
        a.camera_wait = max(a.camera_wait, CAMERA_REBOOT_S)
    cam = CameraStream(a.camera_url, max_age_s=a.max_frame_age).start()
    view = (LensView(a.lens_hfov, size=VIEW_SIZE, view_hfov_deg=TRAINING_HFOV_DEG)
            if a.lens_hfov else None)
    hold = TiltHold(gimbal, a.tilt_deg).start() if gimbal is not None and a.gimbal else None
    return ActionSequenceSource(cam, client, goal, move_s=a.move_seconds, tilt_deg=a.tilt_deg,
                                view=view, hold=hold), cam


CAMERA_REBOOT_S = 120.0     # the stream came back 56 s after a camera restart on the bench


def use_full_hd(gimbal, kbps) -> bool:
    """Ask for 1080p unless it is already set. Returns True if the camera is restarting.

    The camera only serves the new size after a restart: before one, the stream answered
    with no picture at all.
    """
    stream = gimbal.main_stream()
    if stream is None or stream["width"] == 1920:
        print(f"[pilot] camera stream: {stream}")
        return False
    if not gimbal.set_main_stream(1920, 1080, kbps):
        print(f"[pilot] the camera refused 1080p; staying at {stream['width']}x{stream['height']}")
        return False
    restarting = gimbal.reboot_camera()
    print(f"[pilot] camera set to 1080p; restarting it for that to take effect: {restarting}")
    return restarting


async def wait_for_camera_aim(hold, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hold.settled():
            return True
        await asyncio.sleep(0.1)
    return False


def open_tunnel(a):
    """Reach the server by itself, if its login is saved and the URL points at this machine."""
    url = urlparse(a.policy_url)
    if not a.server_tunnel or url.hostname not in ("127.0.0.1", "localhost"):
        return None
    server = load_server()
    if server is None:
        print("[pilot] no saved server login; expecting an ssh tunnel of your own")
        return None
    try:
        tunnel = ServerTunnel(server, url.port).start()
    except OSError:
        print(f"[pilot] port {url.port} is taken; assuming that is your own ssh tunnel")
        return None
    print(f"[pilot] connecting port {url.port} to {server['host']} as requests come")
    return tunnel


def build_driver(a):
    if a.dry_run:
        return DryRunDriver()
    return BenchDriver(a.url) if a.bench else OffboardDriver(a.url)


async def wait_for_frame(cam, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cam.fresh_frame() is not None:
            return True
        await asyncio.sleep(0.1)
    return False


async def main_async(a):
    sup = Supervisor(SafetyLimits(max_horizontal_speed=a.speed,
                                  max_vertical_speed=a.max_vertical_speed,
                                  max_yaw_rate=a.max_yaw_rate,
                                  max_altitude=a.max_altitude,
                                  max_radius=a.max_radius))
    grounded = a.bench or a.dry_run
    driver = build_driver(a)
    source, cam = build_source(a)
    sequence = source if isinstance(source, ActionSequenceSource) else None
    client = getattr(source, "client", None)
    tunnel = open_tunnel(a) if sequence is not None else None
    gate = None
    if a.approve or sequence is not None:
        # Nothing the policy proposes reaches the aircraft until someone clicks Run,
        # so this only makes sense with the panel open.
        # An action waits for its answer: the aircraft is holding still, so the view it was
        # computed from does not go stale the way it does in flight.
        offer_s = float("inf") if sequence is not None else DEFAULT_OFFER_S
        gate = ApprovalGate(source, limits=sup.limits, hold_s=a.move_seconds, offer_s=offer_s)
        source = gate
    ui = uisrv = None
    if a.ui or sequence is not None:
        ui = UiState(sup.limits, mode="BENCH" if a.bench else "DRY RUN" if a.dry_run else "")
        wide = LensView(a.lens_hfov) if cam is not None and a.lens_hfov else None
        ui.attach_camera(cam, wide=wide, model=sequence.view if sequence is not None else None)
        if sequence is not None:
            ui.attach_model(sequence, load_shots(sequence.goal))
        uisrv = UiServer(ui, sup, port=a.ui_port, gate=gate).start()
        print(f"[pilot] panel at {uisrv.url}")

    def phase(text):
        print(f"[pilot] {text}")
        if ui is not None:
            ui.preflight(text)

    airborne = False
    try:
        if cam is not None:
            phase(f"Waiting for the first camera frame, up to {a.camera_wait:.0f} s.")
            if not await wait_for_frame(cam, a.camera_wait):
                if not grounded:
                    raise SystemExit(f"no frame from {a.camera_url}; refusing to take off")
                print(f"[pilot] no frame from {a.camera_url}; carrying on, this will not fly")
        if sequence is not None and sequence.hold is not None:
            phase(f"Aiming the camera at {sequence.hold.target:.0f}° from the horizon.")
            if not await wait_for_camera_aim(sequence.hold, a.camera_wait):
                why = (f"the camera reports {sequence.hold.tilt} and should be at "
                       f"{sequence.hold.target:.0f}°")
                if not grounded:
                    raise SystemExit(f"{why}; refusing to take off")
                print(f"[pilot] {why}; carrying on, this will not fly")
        if sequence is not None:
            phase("Asking the server for one forward pass. The first one is slow.")
            if not await asyncio.to_thread(sequence.check, a.policy_warmup):
                why = f"no actions from {a.policy_url}: {client.last_error}"
                if tunnel is not None and tunnel.last_error:
                    why += f" (server connection: {tunnel.last_error})"
                if not grounded:
                    raise SystemExit(f"{why}; refusing to take off")
                print(f"[pilot] {why}; carrying on, this will not fly")
        phase(f"Connecting to {a.url}.")
        await driver.connect()
        phase("Waiting for a usable position estimate.")
        await driver.wait_until_ready()
        if sup.stats["estopped"]:
            raise SystemExit("the stop was pressed on the ground; refusing to take off")

        airborne = not grounded
        phase(f"Arming, taking off to {a.altitude} m." if airborne else "Not arming.")
        await driver.arm_and_takeoff(a.altitude)
        if airborne:
            await asyncio.sleep(a.takeoff_settle)
        await driver.start()
        print(f"[pilot] offboard engaged; source={a.source}" if airborne
              else f"[pilot] grounded; source={a.source}")
        stats = await run(driver, sup, source, policy_hz=a.policy_hz, duration_s=a.duration,
                          ui=ui, camera=cam, client=client, gate=gate)
        print(f"[pilot] {stats}")
        if cam is not None:
            print(f"[pilot] camera {cam.stats}")
    except KeyboardInterrupt:
        print("\n[pilot] Ctrl-C -> e-stop")
        sup.estop()
    finally:
        if airborne:
            print("[pilot] landing")
        await driver.stop_and_land()
        if cam is not None:
            cam.stop()
        if sequence is not None and sequence.hold is not None:
            sequence.hold.stop()
        if uisrv is not None:
            uisrv.stop()
        if tunnel is not None:
            tunnel.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="udpin://0.0.0.0:14540",
                   help="PX4 connection (SITL default). A radio without a router: "
                        "serial://COM5:57600")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="no aircraft and no MAVSDK; try the panel anywhere")
    p.add_argument("--bench", action="store_true",
                   help="real telemetry, never arms and sends nothing; for props-off tests")
    p.add_argument("--source", default="square", choices=["square", "hover", "policy"])
    p.add_argument("--policy-url", default="http://127.0.0.1:8777/act", dest="policy_url")
    p.add_argument("--no-server-tunnel", action="store_false", dest="server_tunnel",
                   help="do not connect to the server with the login saved by "
                        "scripts/save_server_login.py")
    p.add_argument("--policy-timeout", type=float, default=10.0, dest="policy_timeout")
    p.add_argument("--policy-warmup", type=float, default=120.0, dest="policy_warmup",
                   help="timeout for the forward pass on the ground; the first one is slow")
    p.add_argument("--policy-hz", type=float, default=2.0, dest="policy_hz")
    p.add_argument("--altitude", type=float, default=5.0)
    p.add_argument("--takeoff-settle", type=float, default=8.0, dest="takeoff_settle")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--max-altitude", type=float, default=30.0, dest="max_altitude")
    p.add_argument("--max-radius", type=float, default=50.0, dest="max_radius")
    p.add_argument("--max-vertical-speed", type=float, default=1.0, dest="max_vertical_speed")
    p.add_argument("--max-yaw-rate", type=float, default=60.0, dest="max_yaw_rate")
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--camera-wait", type=float, default=15.0, dest="camera_wait",
                   help="how long to wait for a first frame before refusing to take off")
    p.add_argument("--camera-url", default=None, dest="camera_url",
                   help="RTSP/UDP URL of the camera on its own wireless link, "
                        "e.g. rtsp://192.168.1.50:554/live")
    p.add_argument("--max-frame-age", type=float, default=0.7, dest="max_frame_age",
                   help="older than this and the policy is not asked -> hover")
    p.add_argument("--goal-file", default=None, dest="goal_file",
                   help="the shot to ask for: see configs/drone_goal_example.json")
    p.add_argument("--lens-hfov", type=float, default=A2_MINI_HFOV_DEG, dest="lens_hfov",
                   help="how wide the camera sees, in degrees; scripts/check_lens.py measures it. "
                        "0 sends frames as they are, for a source that is already narrow")
    p.add_argument("--tilt-deg", type=float, default=0.0, dest="tilt_deg",
                   help="where the camera is set, from level, up+; the actions depend on it. "
                        "With --gimbal, where it starts")
    p.add_argument("--gimbal", action="store_true",
                   help="aim the camera as each action is run, and hold that angle from the "
                        "horizon as the aircraft tips. The camera will move")
    p.add_argument("--no-full-hd", action="store_false", dest="full_hd",
                   help="do not ask the camera for a 1080p stream")
    p.add_argument("--stream-kbps", type=int, default=4096, dest="stream_kbps")
    p.add_argument("--approve", action="store_true",
                   help="fly move by move, you click Run in the panel; --source policy always does")
    p.add_argument("--move-seconds", type=float, default=1.5, dest="move_seconds",
                   help="how long one accepted move flies before the aircraft holds")
    p.add_argument("--ui", action="store_true", help="serve the ground-station panel on localhost")
    p.add_argument("--ui-port", type=int, default=8080, dest="ui_port")
    a = p.parse_args()
    if a.bench and a.dry_run:
        p.error("--bench and --dry-run are different tests; pick one")
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\n[pilot] interrupted")


if __name__ == "__main__":
    main()
