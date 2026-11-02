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

from src.drone_control.loop import run
from src.drone_control.offboard import OffboardDriver
from src.drone_control.policy_client import PolicyClient
from src.drone_control.safety import SafetyLimits
from src.drone_control.sources import HoverSource, SquareSource
from src.drone_control.supervisor import Supervisor


def build_source(a):
    if a.source == "square":
        return SquareSource(speed=a.speed)
    if a.source == "hover":
        return HoverSource()
    return PolicyClient(url=a.policy_url, timeout_s=a.policy_timeout)


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
    try:
        stats = await run(driver, sup, build_source(a),
                          policy_hz=a.policy_hz, duration_s=a.duration)
        print(f"[pilot] {stats}")
    except KeyboardInterrupt:
        print("\n[pilot] Ctrl-C -> e-stop")
        sup.estop()
    finally:
        print("[pilot] landing")
        await driver.stop_and_land()


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
    a = p.parse_args()
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\n[pilot] interrupted before offboard; nothing to land")


if __name__ == "__main__":
    main()
