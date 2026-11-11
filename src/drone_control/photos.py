"""Taking the photo: the camera's own frame, saved on the laptop with what it was taken for.

SIYI lists the A2 mini's photo command as unsupported, so a photo is a frame from the video
stream, at the stream's full size (1920x1080 once it is set to Full HD). Next to it go the views
the panel shows, and a JSON note of the shot asked for and where the aircraft and camera were.

The operator takes it. The policy's own "arrived" channel rarely fires (once in six evaluation
episodes that ended within its chunk), so it is shown, not trusted to press the shutter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .camera import encode_jpeg

QUALITY = 95


class PhotoTaker:
    def __init__(self, camera, out_dir, views: dict | None = None, clock=time.time):
        self.camera = camera
        self.out_dir = Path(out_dir)
        self.views = {k: v for k, v in (views or {}).items() if v is not None}
        self._clock = clock
        self.taken: list[dict] = []

    def take(self, meta: dict | None = None) -> dict | None:
        """Save the newest frame. `None` if there is no frame to save."""
        frame = self.camera.latest() if self.camera is not None else None
        if frame is None:
            return None
        now = self._clock()
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for which, image in [("photo", frame.image)] + [(k, v(frame.image)) for k, v in self.views.items()]:
            jpeg = encode_jpeg(image, QUALITY)
            if jpeg is None:
                continue
            path = self.out_dir / (f"{name}.jpg" if which == "photo" else f"{name}_{which}.jpg")
            path.write_bytes(jpeg)
            files[which] = path.name
        if "photo" not in files:
            return None
        shape = getattr(frame.image, "shape", None)
        record = {"name": name, "wall": now, "dir": str(self.out_dir.resolve()), "files": files,
                  "size": [int(shape[1]), int(shape[0])] if shape is not None else None,
                  "frame_seq": frame.seq, "frame_age_s": frame.age_s()}
        (self.out_dir / f"{name}.json").write_text(
            json.dumps({**record, **(meta or {})}, indent=1, default=str), encoding="utf-8")
        self.taken.append(record)
        return record

    def path(self, name: str, which: str = "photo") -> Path | None:
        """A saved file, only for photos taken here: the panel cannot name any other file."""
        for record in self.taken:
            if record["name"] == name and which in record["files"]:
                return self.out_dir / record["files"][which]
        return None

    def summary(self) -> dict:
        return {"count": len(self.taken), "last": self.taken[-1] if self.taken else None,
                "dir": str(self.out_dir.resolve())}


__all__ = ["PhotoTaker"]
