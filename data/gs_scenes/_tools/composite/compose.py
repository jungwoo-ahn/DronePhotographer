"""Composite Blender subject renders over splat backgrounds with a per-pixel depth test.

  compose.py JOBS.json [JOBS.json ...]

A subject pixel survives where it is nearer than the splat surface (15 cm slack, so feet on
uneven splat ground do not flicker) or where the splat is empty; the shadow catcher's shadow
goes through the same test, so a bench in front of the subject hides its shadow too.
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import OpenEXR  # noqa: E402  (this opencv build has no EXR codec)
from PIL import Image  # noqa: E402

SLACK_M = 0.15
for jf in sys.argv[1:]:
    for job in json.loads(Path(jf).read_text()):
        if not Path(job["fg"]).exists():
            print("missing fg", job["fg"]); continue
        fg = cv2.imread(job["fg"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        fg = fg[..., [2, 1, 0, 3]]                                     # BGRA -> RGBA
        exr = sorted(glob.glob(str(Path(job["fg_depth"]).with_suffix("")) + "_*.exr"))
        fgd = next(iter(OpenEXR.File(exr[-1]).channels().values())).pixels
        fgd = fgd[..., 0] if fgd.ndim == 3 else fgd             # Z written into R (=G=B)
        bg = np.asarray(Image.open(job["bg"]).convert("RGB")).astype(np.float32) / 255.0
        bgd = np.load(job["bg_depth"]); bga = np.load(job["bg_alpha"]).astype(np.float32)
        a = fg[..., 3]
        visible = (fgd < bgd + SLACK_M) | (bga < 0.5)
        a = a * visible
        out = bg * (1 - a[..., None]) + fg[..., :3] * a[..., None]
        Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8)).save(job["out"], quality=90)
        print("composited", job["out"])
