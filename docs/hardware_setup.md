# Connecting the real drone

The laptop talks to three things over three separate links:

```
PX4 drone    <-- telemetry radio (USB, a COM port) -->  laptop
SIYI A2 mini <-- Ethernet --> TP-Link <-- Wi-Fi -->     laptop
GPU server   <-- SSH tunnel over a second network -->   laptop
```

Run the pilot **on Windows, not in WSL**: WSL cannot see the radio's COM port. WSL is only
for the simulator (`docs/laptop_setup.md`).

**Nothing here has been tried on the hardware yet.** Camera and router details come from
the vendor documents; the commands were written without the devices attached.

## 1. Laptop

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install "mavsdk<4" opencv-python requests
python scripts/run_sitl_pilot.py --dry-run --approve --ui --duration 600 --takeoff-settle 0
```

Open `http://127.0.0.1:8080`. The band says `DRY RUN`, and no aircraft or MAVSDK is
involved. Enter runs the move on offer, Esc skips it and the space bar stops.

## 2. Router and camera

The A2 mini's address is fixed by default at `192.168.144.25`, so the Wi-Fi network has to
be in `192.168.144.x`. It has Ethernet but no Wi-Fi of its own, so the TP-Link bridges it.

**Router (TL-WR802N), on the desk.** It has one Ethernet port, 2.4 GHz Wi-Fi and no mode
switch, so the mode is set in its web page:

1. Reset it, then join its default Wi-Fi using the details on its label.
2. Log in at `http://192.168.0.254` or `http://tplinkwifi.net` with `admin` / `admin`.
3. **Quick Setup** > **Access Point**. In Router mode the single Ethernet port is usually
   the WAN port, and the camera would be cut off from the Wi-Fi.
4. Set your own network name and password.
5. When it asks for the LAN IP type, do not keep **Smart IP**, which expects another router
   to hand out addresses. Choose **Static** with `192.168.144.1`, mask `255.255.255.0`.
   TP-Link's guide does not show this screen, so check it the first time.
6. Reboot. The router's page is now at `http://192.168.144.1`.

**Camera:**

1. It needs **12 V** (average 2 W, peak 12 W) from the drone. It does not run from the
   router's USB supply.
2. Its network port is GH 1.25 mm, so it needs an adapter cable to RJ45. SIYI's warning:
   RX+ goes to RX+ and RX- to RX-. Check the pinout in the manual before powering up.
3. The stream is `rtsp://192.168.144.25:8554/main.264`.

**Laptop, as administrator:**

```powershell
netsh interface ip set address "Wi-Fi" static 192.168.144.30 255.255.255.0
ping 192.168.144.25
```

Leave out the gateway, so internet traffic does not try to go through the camera network.
This setting stays on the adapter; undo it with
`netsh interface ip set address "Wi-Fi" dhcp`. If the ping fails, check the Windows firewall
and the cable pinout.

Mount the router away from the GPS and the telemetry antenna.

**The lens.** `src/drone_control/lens.py` takes the wide-angle bend out of each frame before it
goes to the server, and cuts it to the view the renders have: 29.86° by 22.62°, 4:3, sent at 512x384.
The server squashes that to its square, as in training. At the training distances of about 4 m this
view holds a whole person, but on a desk it looks very zoomed in. The panel still shows the camera's
own picture.

```powershell
python scripts/check_lens.py --out lens_check   # compare wide.jpg with view.jpg
python scripts/check_lens.py --measure          # the camera tilts; gives the real --lens-hfov
```

The pilot's default is 140°, measured on our camera in the middle of the picture (136 to 141 over
four runs), rather than the data sheet's 160°. Measure again if the camera or its stream settings
change, and pass the answer as `--lens-hfov`. Use `--lens-hfov 0` for a source that is already
narrow, such as a clip of renders.

The panel shows the widest straight 4:3 view, the frame's full height, beside what the model sees.
The lens curve at the edges is an estimate from the data sheet, so expect some bend left there. The
SIYI FPV app's Gimbal Camera page also has a distortion correction switch. Note how it is set, and
measure the lens again if it changes.

**Full HD.** At 1280x720 the model's view holds about 273x207 real pixels, and at 1080p about
410x310. For `--source policy`, the pilot asks for 1080p at 4 Mbps at startup, which `--no-full-hd`
turns off, and `python scripts/check_gimbal.py --main-stream 1080` asks by itself. SIYI's SDK
support table does not list the command for the A2 mini, but ours accepted it. The stream then
served no picture until the camera restarted, so both restart the camera (not the gimbal)
afterwards. The picture is back in about a minute, at 1920x1080 and 30 fps, and the setting stayed
through the restart. If the camera ever refuses, the SIYI FPV app's Video Resolution setting does
the same. Measure the lens again at 1080p.

**Holding an angle.** SIYI lists no lock mode for the A2 mini, and on the bench it stayed still
relative to the airframe after a set angle. `--gimbal` holds the angle from the laptop, steering on
the angle the gimbal reports, and moves the camera as each action is run. While the camera is away
from where the next action assumes, nothing new is offered and Run does nothing. It corrects a few
times a second, so it takes out slow tipping, not vibration. Check this first:

- [ ] `--dry-run --gimbal` with the drone in your hands: tip it 20°, and the camera comes back.
- [ ] Turn it past where the gimbal can reach: Run does nothing, and the camera does not thrash.

At its limits and upside down the gimbal reports angles that are not real, and `TiltHold` ignores
those. The manual says it switches to an upside down mode by itself when laid over.

## 3. Telemetry radio

One COM port can only be open in one program, and both QGroundControl and the pilot need
it, so put [mavp2p](https://github.com/bluenviron/mavp2p) in front. Find the port in Device
Manager, and start mavp2p **before** QGroundControl, or QGC will take the radio for itself:

```powershell
mavp2p serial:COM5:57600 udpc:127.0.0.1:14550 udpc:127.0.0.1:14540
```

QGroundControl listens on 14550 and the pilot's default URL listens on 14540. If QGC still
grabs the radio, turn off its SiK auto-connect in the application settings. 57600 is the
usual SiK rate; radio, PX4 and this command all have to agree.

## 4. Server

The laptop's only Wi-Fi is now on the camera network, so the server needs a second route:
the USB Ethernet adapter or a tethered phone.

The server's repo checkout does not have the drone code, and it holds uncommitted work, so
copy the one file rather than pulling:

```powershell
scp -P 32603 scripts/serve_policy.py <user>@ssh-nipagpu.kakaocloud.com:~/serve_policy.py
ssh -p 32603 <user>@ssh-nipagpu.kakaocloud.com python3 serve_policy.py --port 8777
ssh -p 32603 -N -L 8777:localhost:8777 <user>@ssh-nipagpu.kakaocloud.com
```

That serves the **stub**, which sends a slow left orbit whatever the camera shows. It tests
the connection and the panel, not the model. The real one is
`scripts/serve_cosmos_policy.sh`, from a checkout set up as in `docs/cosmos_checkpoint.md`.

To skip the `ssh -L` window, save the login once. It asks for the password, checks it and keeps it
in Windows Credential Manager, along with the server's host key:

```powershell
pip install paramiko keyring
python scripts/save_server_login.py --host ssh-nipagpu.kakaocloud.com --port 32603 --user <user>
```

From then on, `--source policy` connects its `--policy-url` port to the same port on the server by
itself. This only connects: the model still has to be running there. If the server's host key ever
changes, it refuses to send the password, and saving the login again is how to accept the new key.

## The panel with the model

With `--source policy` the panel at `http://127.0.0.1:8080` also shows:

- **The shot:** pick what the model is asked for. The list holds the goal file, then the 64
  evaluation goals, eight from each side of the subject. Picking one takes back the move on offer
  and asks the model again from the current view. Only the goal changes, sent with each request
  as before; nothing changes on the server. Below the picker are the sentence the model gets, a
  sketch of the side and height to shoot from, and a dashed box over "what the model sees" where
  the shot wants the subject.
- **Where it is going:** whether the model is thinking (a forward pass took about 3 s on the
  server), and its 8 moves drawn from above from where it was asked, with the camera angle after
  each. The moves are only offered one at a time, as before.

Indoors, with no GPS, try it with `--dry-run`: the real camera, gimbal and model, with the
aircraft simulated on the laptop.

```powershell
python scripts/run_sitl_pilot.py --dry-run --source policy --gimbal `
    --camera-url rtsp://192.168.144.25:8554/main.264 `
    --goal-file configs/drone_goal_example.json --duration 1800
```

## 5. Bench test, propellers off

`--bench` reads real telemetry but never arms and sends nothing:

```powershell
python scripts/run_sitl_pilot.py --bench --source policy `
    --camera-url rtsp://192.168.144.25:8554/main.264 `
    --goal-file configs/drone_goal_example.json --tilt-deg -10 --duration 600
```

- [ ] The panel's MAVLink, video and policy lights are all up.
- [ ] Altitude and position match QGroundControl.
- [ ] The camera picture is live; note the frame age in the panel.
- [ ] MAVLink stays "receiving" for several minutes. Flicker means the radio is delivering
      position more slowly than the 1 s limit, and any flight would stutter into holds.
- [ ] Unplug the camera's Ethernet: video goes down and the panel says the signal is lost.
- [ ] Stop the tunnel: the policy link goes down.
- [ ] Unplug the radio: MAVLink goes down, and the space bar still ends the run.

## 6. First flight

Do the simulator's five checks first (`docs/laptop_setup.md`, step 2). Then, in an open
area, with a pilot holding an RC transmitter and QGroundControl open:

```powershell
python scripts/run_sitl_pilot.py --source hover --ui --duration 60 `
    --altitude 2.5 --max-altitude 4 --max-radius 5 `
    --speed 0.3 --max-vertical-speed 0.3 --max-yaw-rate 15
```

It waits for a GPS fix, so it will not take off indoors. Check PX4's own failsafes in QGC
(`COM_OF_LOSS_T`, `COM_OBL_ACT` and the RC and data-link loss actions) before this flight:
the pilot process cannot help once the radio is gone. The first move-by-move flight comes
only after this one is clean.

## Not built yet

**Tilt.** The policy does output it: with roll removed, the rotation in its action splits
into airframe yaw and gimbal tilt, and `src/drone_control/actions.py` does that split. For
now the panel only shows the tilt it wants, and the camera stays at `--tilt-deg`. Measured
over 63,008 training frames, roll is exactly 0 and 1.56% of frames aim above the A2 mini's
+25° limit.

`src/drone_control/gimbal.py` drives it over the SIYI SDK (`192.168.144.25`, port 37260), but
is only tested against a fake so far, and flying does not call it yet. Try it on the bench:

```powershell
python scripts/check_gimbal.py --tilt -30    # the camera moves, and comes back to centre
python scripts/check_gimbal.py --watch 30    # tip the aircraft by hand, propellers off
```

The second answers what the manual does not: if the angle stays put, the gimbal reports
against the horizon, which is what the conversion needs. If it follows, it reports against
the airframe, and the aircraft's pitch has to be taken into account.

Still to build: calling the driver from an approved action, and saving the picture when the
policy says it has arrived. The manual lists the A2 mini's photo command as unsupported, so
that means a frame from the stream.

The A2 mini only stabilises tilt, so the picture rolls with the airframe when it banks, and
the training data has no roll at all. Keep sideways moves gentle.

Vendor documents: [A2 mini manual](https://www.worldronemarket.com/siyi-a2-mini-user-manual/),
[A2 mini FAQ](https://siyi.biz/en/product/single-axis-agri-camera/a2-mini/faq/),
[TP-Link Access Point mode](https://www.tp-link.com/support/faq/1503),
[PX4 telemetry radios](https://docs.px4.io/v1.13/en/telemetry/).
