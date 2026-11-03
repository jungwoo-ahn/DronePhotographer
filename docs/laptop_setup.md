# Setting the laptop up to fly (Windows)

This page is the simulator. For the real drone, radio and camera, see
`docs/hardware_setup.md`.

The laptop is the ground station. It runs PX4 SITL, QGroundControl, the control chain
and the panel. The GPU server runs the policy, and the two meet over one SSH tunnel.

**PX4 does not run natively on Windows**, and that is fine -- PX4's own supported
Windows path is WSL2, a real Ubuntu on the same machine. So the shape of the day is:

```
Windows                         │  WSL2 (Ubuntu)
  QGroundControl (.exe)         │    PX4 SITL + Gazebo
  your browser -> the panel     │    run_sitl_pilot.py
                                │    ssh tunnel to the GPU server
        both sides share localhost, via mirrored networking
```

**The first session needs nothing from the GPU server.** `src/drone_control` imports
with nothing but the standard library (no torch, no OpenCV, no Blender, no Cosmos), the
stub policy runs anywhere, and `--source square` needs no camera and no policy at all.
So get a square flying first and add the server afterwards. Do **not** copy the
checkpoint to the laptop: it is 91 GB, and it needs a 48 GB GPU to be of any use.

## Step 0: WSL2, once

In PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Then create `C:\Users\<you>\.wslconfig` with exactly this:

```ini
[wsl2]
networkingMode=mirrored
```

and apply it with `wsl --shutdown` (it restarts on next use).

That one setting is worth more than it looks. By default WSL2 sits behind NAT with its
own IP address that changes on reboot, and PX4's own documentation says that in that
mode you will not be able to monitor the simulation from QGroundControl on Windows.
Mirrored mode makes `localhost` mean the same thing on both sides, which fixes
QGroundControl, the panel in your browser and the SSH tunnel in one go. It needs
**Windows 11 22H2 or newer**; check with `wsl --version`.

If you are on Windows 10, mirrored mode is not available and neither is WSLg (the bit
that puts Linux windows on your desktop). That is a rougher road: run QGroundControl
*inside* WSL instead of on Windows, and expect to set up an X server for the Gazebo
window. Everything else below is unchanged. If you have the choice, do this on
Windows 11.

## Step 1: PX4, inside WSL

Open the Ubuntu terminal and let the setup script do the work:

```bash
sudo apt update
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
```

Then `wsl --shutdown` from PowerShell and reopen Ubuntu, because the script changes
group membership and the shell has to be restarted to notice.

```bash
cd PX4-Autopilot && make px4_sitl gz_x500
```

The first build is long -- tens of minutes is normal, and it is compiling, not stuck.
The Gazebo window appears on your Windows desktop through WSLg. If it is sluggish or
you just do not want it, `HEADLESS=1 make px4_sitl gz_x500` skips the 3D view; the
simulation and all of MAVLink carry on exactly the same.

## Step 2: a square, with nothing else running

```bash
git clone https://github.com/jungwoo-ahn/DronePhotographer.git
cd DronePhotographer
python3 -m venv .venv && source .venv/bin/activate
pip install "mavsdk<4"

# PX4 from step 1 is still running in its own terminal; in a second Ubuntu terminal:
python scripts/run_sitl_pilot.py --source square --duration 40 --ui
```

Keep the clone in the WSL home directory (`~/DronePhotographer`), **not** under
`/mnt/c/...`. Crossing the Windows filesystem boundary makes git and Python startup
several times slower for no benefit.

Now open `http://localhost:8080` in your normal Windows browser. With mirrored
networking that works directly, and the server still binds loopback only -- which is
deliberate, because that page can stop the aircraft and has no business being reachable
from the drone's own wireless network.

What you are checking, in order:

1. **It arms, takes off and holds.** If offboard is refused, the setpoint stream was
   not running before the mode change; the driver starts it first on purpose, so this
   failing means the connection URL is wrong.
2. **The square is a square.** A sign error shows up as a visibly wrong shape, which
   is the whole reason the stub source is a square rather than a drift.
3. **The panel agrees with QGroundControl** (step 3). Altitude and position match.
4. **The stop works.** Press space in the browser. The aircraft should hover
   immediately and the panel should latch.
5. **Pull the plug on the pilot.** Ctrl-C is a clean e-stop (hover, leave offboard,
   land). Then do it the rude way: in a third Ubuntu terminal, `pkill -9 -f
   run_sitl_pilot`, and watch what PX4 does. MAVSDK keeps resending the last setpoint
   after our process is gone, so the aircraft does **not** stop. That is the failure
   mode the TTL exists for, and it is worth seeing once with your own eyes.

If all five pass, the control chain is real and everything after this is plumbing.

## Step 3: QGroundControl, on Windows

Install the normal Windows build (there is a `.exe` installer) and just start it. It
takes UDP 14550, MAVSDK takes 14540, and they coexist on SITL without a router.

Keep it open as the safety pilot. If anything looks wrong, switch to **Hold** in QGC
rather than trying to Ctrl-C in time: QGC outranks this software, which is drawn in the
panel's own chain of order.

If QGC says *Disconnected* while PX4 is clearly running, it is the networking, not the
simulator. In order: confirm `networkingMode=mirrored` is in `.wslconfig`, confirm you
ran `wsl --shutdown` after writing it, and confirm you clicked **Allow** on the Windows
Firewall prompt (if you dismissed it, allow QGroundControl on private networks by hand).
If you are stuck, installing the QGC AppImage inside WSL sidesteps the issue entirely.

On real hardware there is one telemetry link, not two, so QGC and MAVSDK both need
`mavlink-router` or `mavp2p` in front of them. SITL does not.

## Step 4: the policy, over the tunnel

Run the tunnel from **inside WSL**, in the same place as the pilot, so that
`127.0.0.1` means one thing:

```bash
# on the server
python scripts/serve_policy.py --port 8777

# in WSL
ssh -N -L 8777:localhost:8777 <user>@<server>
python scripts/run_sitl_pilot.py --source policy \
    --policy-url http://127.0.0.1:8777/act \
    --camera-url <see below> --approve --ui
```

`serve_policy.py` is still a stub (a slow left orbit) and says so; the real path raises
rather than returning stub values quietly. It is enough to prove the tunnel, the JSON
contract and the approval flow.

With `--approve`, nothing reaches the aircraft until you click **Run this move** in the
panel. That is the right way to fly a learned policy the first time.

## The camera, which SITL does not give you

`--source policy` is refused without `--camera-url`, by design: a vision policy flown
blind is worse than one that does not fly. SITL has no camera on the plain `gz_x500`
airframe, and Windows adds a wrinkle -- **a USB webcam plugged into the laptop is not
visible inside WSL2**, because WSL2 does not pass USB devices through. So, cheapest
first:

1. **A recorded video file.** `--camera-url ~/clip.mp4` works, because the camera falls
   back to a plain `cv2.VideoCapture` when the GStreamer pipeline cannot open the URL
   (`pip install opencv-python` first). Copy the clip into the WSL home directory.
   Caveat: a file is read as fast as the reader thread can pull frames rather than at
   its own frame rate, so it proves the pipe but is not a realistic feed.
2. **A real camera on its own wireless link.** On Windows this is the *best* option as
   well as the deployment one, because an RTSP stream is just network traffic and WSL2
   handles it with no USB involved: the router the drone carries, RTSP into
   `--camera-url`, nothing through MAVLink.
3. **A camera-equipped airframe.** PX4 ships one (check
   `make px4_sitl list_config_targets`); its images come out on a Gazebo topic, so it
   still needs a bridge to RTSP or to a `v4l2loopback` device.
4. **usbipd-win**, which attaches a USB device into WSL. It exists and works for many
   devices, but webcams usually need a WSL kernel rebuilt with the UVC driver, so it is
   the most expensive option here rather than the obvious one.

A webcam index (`--camera-url 0`) does **not** work as written: the URL is passed to
OpenCV as a string, and OpenCV needs an int for a device. That is a three-line change
if it ever becomes the easiest camera to test with -- but on Windows, options 1 and 2
get there sooner.

## What is not done yet

- The real policy is served by `scripts/serve_cosmos_policy.sh` on the server; see
  `docs/cosmos_checkpoint.md`. The checkpoint is **not** committed and never will be.
- Nothing in this package has ever touched a real aircraft, or PX4 at all. Every test
  runs against fakes. Step 2 above is the first contact with reality, and it should be
  treated that way.
