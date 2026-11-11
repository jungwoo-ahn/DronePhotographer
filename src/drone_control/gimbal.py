"""SIYI A2 mini gimbal, over the SIYI SDK on the camera's network.

A separate link again: UDP to the camera, through the same router as the video. It only
ever aims the camera, so it stays outside the supervisor's chain of order.

The A2 mini has one axis. Yaw commands do nothing on it, and the airframe does the yawing.
Its manual also lists the photo command as unsupported, so there is none here.

Everything answers `None` when the gimbal does not: an unplugged camera must not raise into
whatever is flying. A call can wait up to `timeout_s`, so keep it off the 20 Hz fast lane.

SIYI's documents list no lock, follow or FPV mode for this camera, and on the bench it kept
still relative to the airframe after a set angle while its reported angle changed. So `TiltHold`
does the locking from here, by steering on the angle it reports, which SIYI defines in
North-East-Down coordinates.

**Not yet confirmed on the hardware:** `TiltHold` itself, and what a set angle is measured from.
It should hold either way: if the gimbal already holds the horizon, the error stays near zero.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

from .actions import TILT_MAX_DEG, TILT_MIN_DEG

DEFAULT_HOST = "192.168.144.25"
DEFAULT_PORT = 37260

STX = b"\x55\x66"
NEED_ACK = 1            # the CTRL byte in every example frame in the manual
HEADER = struct.Struct("<2sBHHB")     # STX, CTRL, data length, sequence, command

CMD_CENTER = 0x08
CMD_GET_ATTITUDE = 0x0D
CMD_SET_ANGLE = 0x0E
CMD_GET_ENCODING = 0x20
CMD_SET_ENCODING = 0x21
CMD_REBOOT = 0x80

STREAM_MAIN = 1
H265 = 2
ENCODING = struct.Struct("<BBHHH")     # stream, codec, width, height, kbps; then a last byte


def crc16(data: bytes) -> int:
    """CRC-16/XMODEM, which reproduces every example frame in the manual."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode(cmd: int, data: bytes = b"", seq: int = 0) -> bytes:
    frame = HEADER.pack(STX, NEED_ACK, len(data), seq, cmd) + data
    return frame + struct.pack("<H", crc16(frame))


def decode(frame: bytes) -> tuple[int, bytes] | None:
    """`(command, data)`, or `None` for anything that is not one intact frame."""
    if len(frame) < HEADER.size + 2:
        return None
    stx, _ctrl, length, _seq, cmd = HEADER.unpack_from(frame)
    if stx != STX or len(frame) != HEADER.size + length + 2:
        return None
    if struct.unpack_from("<H", frame, len(frame) - 2)[0] != crc16(frame[:-2]):
        return None
    return cmd, frame[HEADER.size:-2]


def _pitch_of(data: bytes) -> float | None:
    """Pitch from an attitude reply, whose first fields are yaw, pitch, roll in 0.1°."""
    if len(data) < 6:
        return None
    return struct.unpack_from("<hhh", data)[1] / 10.0


class SiyiGimbal:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout_s: float = 0.3):
        self.address = (host, port)
        self.timeout_s = float(timeout_s)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def close(self):
        self._sock.close()

    def _ask(self, cmd: int, data: bytes = b"") -> bytes | None:
        """Send one command and wait for its reply, ignoring anything else that arrives."""
        self._seq = (self._seq + 1) & 0xFFFF
        deadline = time.monotonic() + self.timeout_s
        try:
            self._sock.sendto(encode(cmd, data, self._seq), self.address)
            while (left := deadline - time.monotonic()) > 0:
                self._sock.settimeout(left)
                reply = decode(self._sock.recv(1024))
                if reply is not None and reply[0] == cmd:
                    return reply[1]
        except OSError:                 # includes the timeout, and an unreachable network
            pass
        return None

    def tilt(self) -> float | None:
        """Where the camera is pointing now: degrees, up positive."""
        reply = self._ask(CMD_GET_ATTITUDE)
        return None if reply is None else _pitch_of(reply)

    def steady_tilt(self, samples: int = 5, tolerance_deg: float = 1.0,
                    gap_s: float = 0.1) -> float | None:
        """The tilt, only if several readings agree. Use this for anything acted on.

        Seen on the bench: at its lower limit the gimbal mixed a steady -94° with a value
        sweeping through the whole circle, one reply each. Moving also fails this, on purpose.
        """
        readings = []
        for n in range(samples):
            if n:
                time.sleep(gap_s)
            readings.append(self.tilt())
        if None in readings or max(readings) - min(readings) > tolerance_deg:
            return None
        return sum(readings) / samples

    def set_tilt(self, degrees: float) -> float | None:
        """Aim the camera, inside its range. Returns the angle it reports having now.

        That is where it IS when it answers, not where it is going: it takes time to get there.
        """
        target = max(TILT_MIN_DEG, min(TILT_MAX_DEG, float(degrees)))
        reply = self._ask(CMD_SET_ANGLE, struct.pack("<hh", 0, round(target * 10)))
        return None if reply is None else _pitch_of(reply)

    def center(self) -> bool:
        reply = self._ask(CMD_CENTER, b"\x01")
        return bool(reply and reply[0] == 1)

    def main_stream(self) -> dict | None:
        """What the video stream is set to: width, height, kbps and fps."""
        reply = self._ask(CMD_GET_ENCODING, bytes([STREAM_MAIN]))
        if reply is None or len(reply) < ENCODING.size + 1:
            return None
        _stream, _codec, width, height, kbps = ENCODING.unpack_from(reply)
        return {"width": width, "height": height, "kbps": kbps, "fps": reply[ENCODING.size]}

    def set_main_stream(self, width: int, height: int, kbps: int) -> bool:
        """Ask for 1920x1080 or 1280x720. The manual's support table does not list this for the
        A2 mini, though it answers the matching read, so check `main_stream` and the picture.

        On our A2 mini it accepted 1080p and the stream then served no picture at all, until
        `reboot_camera`. After that it streamed 1920x1080 at 30 fps, and kept the setting.
        """
        data = ENCODING.pack(STREAM_MAIN, H265, width, height, kbps) + b"\x00"
        reply = self._ask(CMD_SET_ENCODING, data)
        return bool(reply and len(reply) >= 2 and reply[1] == 1)

    def reboot_camera(self) -> bool:
        """Restart the camera only, not the gimbal. The stream is back after about a minute."""
        reply = self._ask(CMD_REBOOT, bytes([1, 0]))
        return bool(reply and reply[0] == 1)


class TiltHold:
    """Keeps the camera at an angle from the horizon, correcting as the aircraft tips.

    Runs on its own thread, because each reading can wait on the network. It sends a correction
    only when two readings in a row agree and make sense. At its limits and upside down this
    gimbal has reported values that do neither, and steering on one would throw the camera away.
    Without good readings it stays where it is and `tilt` answers `None`.
    """

    PERIOD_S = 0.2
    GAIN = 0.5              # of the error per step: the gimbal is still moving when read again
    DEADBAND_DEG = 0.5
    MAX_STEP_DEG = 10.0
    AGREE_DEG = 5.0
    MARGIN_DEG = 15.0       # a reading this far outside the gimbal's range is not an angle

    def __init__(self, gimbal: SiyiGimbal, target_deg: float = 0.0):
        self.gimbal = gimbal
        self._lock = threading.Lock()
        self._target = self._limit(target_deg)
        self._command = self._target
        self._last: float | None = None
        self._tilt: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _limit(degrees: float) -> float:
        return max(TILT_MIN_DEG, min(TILT_MAX_DEG, float(degrees)))

    @property
    def target(self) -> float:
        with self._lock:
            return self._target

    @property
    def tilt(self) -> float | None:
        """The last angle that could be trusted, or `None` if the last reading could not."""
        with self._lock:
            return self._tilt

    def set_target(self, degrees: float) -> None:
        with self._lock:
            step = self._limit(degrees) - self._target
            self._target += step
            self._command = self._limit(self._command + step)

    def settled(self, tolerance_deg: float = 3.0) -> bool:
        tilt = self.tilt
        return tilt is not None and abs(tilt - self.target) <= tolerance_deg

    def step(self) -> None:
        """One reading and at most one correction."""
        reading = self.gimbal.tilt()
        sane = (reading is not None
                and TILT_MIN_DEG - self.MARGIN_DEG <= reading <= TILT_MAX_DEG + self.MARGIN_DEG)
        agreed = sane and self._last is not None and abs(reading - self._last) <= self.AGREE_DEG
        self._last = reading if sane else None
        with self._lock:
            self._tilt = reading if agreed else None
            if not agreed:
                return
            error = self._target - reading
            if abs(error) <= self.DEADBAND_DEG:
                return
            change = max(-self.MAX_STEP_DEG, min(self.MAX_STEP_DEG, self.GAIN * error))
            self._command = self._limit(self._command + change)
            command = self._command
        self.gimbal.set_tilt(command)

    def start(self) -> "TiltHold":
        self.gimbal.set_tilt(self._command)
        self._thread = threading.Thread(target=self._run, name="tilt-hold", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.PERIOD_S):
            self.step()
