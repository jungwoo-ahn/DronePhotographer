# Drone control (PX4 SITL, velocity setpoints)

`src/drone_control/` is the laptop-side control chain. It is deliberately separate from
everything else in `src/`: it imports no Blender, no torch, no Cosmos, and runs on a
laptop that has none of them. The policy stays on the GPU server behind HTTP.

```
PX4 SITL  ◄──MAVLink UDP──  laptop: supervisor + offboard driver
                                      │  HTTP over an SSH tunnel, ~1-2 Hz
                                      ▼
                            server: scripts/serve_policy.py (GPU)
```

The server **never commands the drone**. It answers "given this frame, what velocity?"
and the laptop decides whether to obey. Authority stays where the failsafe is.

## Two facts that shape the whole design

**PX4 drops out of offboard below 2 Hz.** `COM_OF_LOSS_T` fires and `COM_OBL_ACT`
takes over, and the setpoint stream has to be running *before* offboard is engaged.
A remote policy at 0.5-2 Hz cannot drive PX4 directly, so the loop runs at two rates:
a 20 Hz fast lane that decides and sends, and a slow lane that asks the policy.

**MAVSDK resends the last setpoint at 20 Hz on its own.** That satisfies PX4's
liveness check for free -- and is exactly why this package needs its own watchdog.
PX4 cannot distinguish "the operator wants this velocity" from "the laptop died and
MAVSDK is echoing a corpse". Every command therefore carries a TTL, measured from
when the *laptop* received it (the two machines do not share a clock), and an expired
command becomes a hover.

## The chain of order

Highest priority first, in `supervisor.Supervisor.decide`:

| # | Source | Action |
|---|---|---|
| 1 | E-stop (latched) | hover, then land |
| 2 | Safety envelope | clamp speed / altitude / geofence |
| 3 | Stale command | hover -- the policy link is gone |
| 4 | Policy command | pass through, clamped |
| 5 | Nothing yet | hover |

Only the supervisor talks to `offboard`, and `OffboardDriver.send` takes a `Decision`
rather than a `VelocityCommand` so that routing a raw policy command straight to the
airframe is awkward to write by accident.

A single failed HTTP call is *not* a stop: `submit(None)` is ignored and the previous
command keeps flying until its TTL runs out. Repeated failures stop the drone.

## Conventions: three of four axes flip

Ours (`VelocityCommand`, matching `src/policy/astra/pilot.py`) against MAVSDK's
`VelocityBodyYawspeed`:

| ours | MAVSDK | convert |
|---|---|---|
| `vx` forward+ | `forward_m_s` forward+ | — |
| `vy` **left**+ | `right_m_s` **right**+ | negate |
| `vz` **up**+ | `down_m_s` **down**+ | negate |
| `yaw_rate` **left**+ (CCW) | `yawspeed_deg_s` **clockwise**+ | negate |

All of it lives in `conventions.to_mavsdk_body` and nowhere else. This repo has had an
inverted yaw in three separate places before, so the boundary gets one named function
and its own tests.

## The camera is a separate link

The image does **not** come down the MAVLink channel. The camera sits behind its own
wireless router and streams RTSP/UDP straight to the laptop, so there are now three
links that fail independently:

```
PX4   ──MAVLink UDP 14540──►  laptop  ◄──RTSP/UDP──  camera (own wireless router)
                                 │
                                 └──HTTP over SSH──►  policy server
```

None of them can detect another's failure, and the video link is the nastiest,
because its failure is **invisible**: frames simply stop arriving while the last
decoded image sits in memory looking perfectly valid. Acting on it means flying by a
photograph of where the drone used to be — at 3 m/s, a 2 s-stale frame is 6 m of lie.

So `Frame` carries a receive time and `--max-frame-age` (default 0.7 s) gates it, and
`VisionPolicySource` **refuses to call the policy at all without a fresh frame**. It
returns `None`, the last command ages out through the TTL, and the drone hovers. One
watchdog mechanism covers both remote links instead of two half-watchdogs.

**Latest-frame-only.** OpenCV queues frames internally, so reading sequentially from a
~2 Hz loop hands you an ever-growing backlog — the stream plays in slow motion and
latency climbs without bound. `CameraStream` runs a reader thread flat out and keeps
only the newest frame, deliberately discarding everything in between; the GStreamer
path pushes the same idea down into the pipeline with `drop=true max-buffers=1`.
JPEG encoding happens on demand at policy rate, not at video rate.

## Relationship with QGroundControl

**They coexist — different UDP ports, by design.** PX4 SITL publishes to both:

| Port | Who |
|---|---|
| 14550 | QGroundControl |
| 14540 | offboard APIs (MAVSDK — us) |
| 4560 (TCP) | the simulator itself |

Run QGC and `run_sitl_pilot.py` at the same time; neither needs to know about the
other. On a **real** airframe with a single telemetry radio there is only one link,
and you need `mavlink-router` (or MAVSDK's passthrough) to fan it out to both — that
is a real deployment difference from SITL, not a detail.

QGC is not a passive viewer. It sits **above** this package in the chain of order:

- Switching flight mode in QGC (Hold, Position, RTL) **takes the aircraft out of
  offboard**, and our setpoints stop having any effect. That is the intended
  human override and the real top of the chain — above the software e-stop, which
  can only act if our process is still alive.
- An RC transmitter, where present, outranks QGC in turn.

So the full order is: **RC → QGC → safety envelope → our e-stop → staleness → policy.**
This package's authority begins only after everything above it has declined to
intervene. Use QGC as the safety pilot while bringing the loop up: watch the mode
chip, and if anything looks wrong switch to Hold rather than trying to Ctrl-C in time.

## Running it

```bash
# terminal 1 -- PX4 SITL
make px4_sitl gz_x500

# terminal 2 -- laptop, stub source, no policy needed
pip install mavsdk
python scripts/run_sitl_pilot.py --source square --duration 40

# with the policy server and the camera
python scripts/serve_policy.py --port 8777            # on the server
ssh -N -L 8777:localhost:8777 <user>@<server>         # on the laptop
python scripts/run_sitl_pilot.py --source policy \
    --camera-url rtsp://192.168.1.50:554/live --brief "orbit left, subject centred"
```

`--source policy` without `--camera-url` is refused rather than flown blind.
QGroundControl can stay open throughout; it uses a different port.

Ctrl-C is an e-stop: hover, leave offboard, land -- rather than dropping the link and
letting PX4's failsafe decide.

## The panel

`python scripts/run_sitl_pilot.py --source square --ui` serves a ground station on
`http://127.0.0.1:8080`. It is stdlib-only (no Flask, no node) and binds to loopback,
because the page can stop the aircraft and should not be reachable from the drone's
own wireless network.

It **observes the supervisor and is not another rung in the chain of order**: the only
state-changing routes are `/api/estop` and `/api/clear_estop`, both of which can only
make the aircraft safer. There is deliberately no "set velocity" endpoint, and a test
asserts that asking for one 404s.

What it shows, and why each panel earns its place:

- **Three link pills** — MAVLINK / VIDEO / POLICY, the three independent failure
  domains. Video shows frame age in ms; policy shows calls and failures. This is the
  one view where "which link died?" is answerable at a glance.
- **Chain of order** — the live ladder with the active rung lit, including the two
  rungs *above* this software (RC, QGC) greyed out, so the panel never implies it has
  more authority than it does.
- **Command, in both frames** — ours beside what MAVSDK was handed, with ⇄ on the
  three axes that invert. The bars grow from the centre, so a sign error appears as a
  bar on the wrong side rather than as a crash.
- **Envelope gauges** — speed, altitude, distance from home against their limits,
  amber at 80%.
- **Camera** with a rule-of-thirds reticle (the policy is scored on where the subject
  sits in frame) and a frame-age badge; it says NO SIGNAL and names the consequence
  ("policy not asked") rather than freezing on a stale picture.
- **Event log** — reason transitions only, not 20 Hz of identical lines.

Space bar is the keyboard e-stop; the mouse is the slow path when something is going
wrong. If the pilot process dies the footer says so, because a panel that silently
freezes looks identical to a drone that is holding position.

A UI fault cannot ground the aircraft: `publish()` is wrapped, and a test runs the loop
with a panel that raises on every tick and asserts the drone still flies.

## Status

- Control chain, safety envelope, watchdog, sources, loop: **done**, 28 tests, no
  MAVSDK or network needed to run them.
- `scripts/serve_policy.py` is a **stub** (a slow left orbit). The real path raises
  rather than silently returning stub values.
- Camera ingest on its own link, latest-frame-only, with a staleness gate: **done**,
  tested against a fake capture (no OpenCV or camera needed).
- Ground-station panel (`--ui`): **done**, stdlib-only, 8 tests including "the UI
  offers no way to command the drone" and "a broken UI cannot stop the loop".
- Not yet done: consolidating the cosmos3 checkpoint (8 FSDP `.distcp` shards, 85 GB)
  into a loadable state dict; and flying it, which needs PX4 on the laptop.
