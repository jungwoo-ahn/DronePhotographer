# Setting the laptop up to fly

The laptop is the ground station. It runs PX4 SITL, QGroundControl, the control chain
and the panel. The GPU server runs the policy and holds the checkpoint, and the two
meet over one SSH tunnel.

**The first session needs none of that.** `src/drone_control` imports with nothing but
the standard library (no torch, no OpenCV, no Blender, no Cosmos), the stub policy runs
anywhere, and `--source square` needs no camera and no policy at all. So bring the
laptop up on its own first, and add the server only once a square flies.

## What to install

| | Why | Note |
| --- | --- | --- |
| Python 3.10+ | the control chain | `pip install mavsdk` is the only hard dependency |
| `opencv-python` | the camera link | only for `--source policy`; skip it for the first flight |
| PX4-Autopilot | the simulated aircraft | the long install; follow PX4's own toolchain page |
| QGroundControl | the map, and the safety pilot | AppImage, no install needed |

Nothing else. Do **not** copy the checkpoint to the laptop: it is 91 GB of FSDP shards
that only the server's GPU can load.

## Step 1: a square, with nothing else running

```bash
git clone https://github.com/jungwoo-ahn/DronePhotographer.git
cd DronePhotographer && git checkout 009-mavsdk-drone-control
pip install mavsdk

# terminal 1
cd PX4-Autopilot && make px4_sitl gz_x500

# terminal 2
python scripts/run_sitl_pilot.py --source square --duration 40 --ui
```

Open `http://127.0.0.1:8080`. What you are checking, in order:

1. **It arms, takes off and holds.** If offboard is refused, the setpoint stream was
   not running before the mode change; the driver starts it first on purpose, so this
   failing means the connection URL is wrong.
2. **The square is a square.** A sign error shows up as a visibly wrong shape, which
   is the whole reason the stub source is a square rather than a drift.
3. **The panel agrees with QGroundControl.** Altitude and position should match.
4. **The stop works.** Press space. The aircraft should hover immediately, and the
   panel should latch.
5. **Pull the plug on the pilot.** Ctrl-C is an e-stop (hover, leave offboard, land).
   Separately, kill `-9` the pilot in another terminal and watch what PX4 does: MAVSDK
   keeps resending the last setpoint, so this is the failure mode the TTL exists for.

If all five pass, the control chain is real and everything after this is plumbing.

## Step 2: QGroundControl alongside

Start QGC at any point. It takes port 14550, MAVSDK takes 14540, and they coexist on
SITL without a router. Keep it open as the safety pilot: if anything looks wrong,
switch to Hold in QGC rather than trying to Ctrl-C in time. QGC outranks this software,
which is drawn in the panel's own chain of order.

On real hardware there is one telemetry link, not two, so QGC and MAVSDK both need
`mavlink-router` or `mavp2p` in front of them. SITL does not.

## Step 3: the policy, over the tunnel

```bash
# on the server
python scripts/serve_policy.py --port 8777

# on the laptop
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
airframe, so there are three ways to give it one, cheapest first:

1. **A recorded video file.** `--camera-url /path/to/clip.mp4` works, because the
   camera falls back to a plain `cv2.VideoCapture` when the GStreamer pipeline cannot
   open the URL. Caveat: a file is read as fast as the reader thread can pull frames
   rather than at its own frame rate, so it proves the pipe but is not a realistic feed.
2. **A camera-equipped airframe.** PX4 ships one with a camera (check
   `make px4_sitl list_config_targets`); its images come out on a Gazebo topic, so it
   still needs a bridge to RTSP or to a `v4l2loopback` device.
3. **A real camera on its own wireless link**, which is the deployment setup: the
   router the drone carries, RTSP into `--camera-url`, nothing through MAVLink.

A webcam index (`--camera-url 0`) does **not** work as written: the URL is passed to
OpenCV as a string, and OpenCV needs an int for a device. That is a three-line change
if it turns out to be the easiest camera to test with.

## What is not done yet

- The cosmos3 checkpoint is on the server at
  `checkpoints/camera_policy_nano_v6/iter_000045000/`, verified byte-exact, 91.06 GB.
  It is 8 FSDP `.distcp` shards and still needs consolidating into a loadable state
  dict before `serve_policy.py --checkpoint` can do anything but raise.
- Nothing in this package has ever touched a real aircraft, or PX4 at all. Every test
  runs against fakes. Step 1 above is the first contact with reality, and it should be
  treated that way.
