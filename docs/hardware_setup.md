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
pip install "mavsdk<4" opencv-python
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

**The lens is a real mismatch.** The A2 mini sees 160° across. The renderer's defaults
(`--focal_length 24 --sensor_width 12.8`) give about 30°. Look at this before reading
anything into a real policy's behaviour on this camera.

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

That serves the **stub**, which flies a slow left orbit whatever the camera shows. It tests
the connection and the panel, not the model.

## 5. Bench test, propellers off

`--bench` reads real telemetry but never arms and sends nothing:

```powershell
python scripts/run_sitl_pilot.py --bench --source policy `
    --camera-url rtsp://192.168.144.25:8554/main.264 --approve --ui --duration 600
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

**Tilt.** The policy does output it. Its 10D action is `[Δtranslation_cam(3), rot6d(6),
shoot]`, and with roll removed the rotation splits into airframe yaw and gimbal tilt.
`src/policy/common/action_to_drone.py` does that conversion, and needs the current tilt,
because the translation is in the tilted camera's frame. Measured over 63,008 training
frames, roll is exactly 0 and 1.56% of frames aim above the A2 mini's +25° limit.

Still to build:

- Carry `tilt_rate` through the server reply, `VelocityCommand` and the approval panel.
- A gimbal driver on the SIYI SDK (`192.168.144.25`, port 37260) that sets the tilt and
  reads it back. Check on the hardware whether its angle is relative to the airframe or
  the horizon. The conversion needs the horizon.
- A real duration for one action. The renders have no clock: `dt` sets the speed.
- The real model on the server. `serve_policy.py` is still the stub.

The A2 mini only stabilises tilt, so the picture rolls with the airframe when it banks, and
the training data has no roll at all. Keep sideways moves gentle.

Vendor documents: [A2 mini manual](https://www.worldronemarket.com/siyi-a2-mini-user-manual/),
[A2 mini FAQ](https://siyi.biz/en/product/single-axis-agri-camera/a2-mini/faq/),
[TP-Link Access Point mode](https://www.tp-link.com/support/faq/1503),
[PX4 telemetry radios](https://docs.px4.io/v1.13/en/telemetry/).
