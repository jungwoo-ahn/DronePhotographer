"""Take the wide-angle bend out of the A2 mini's picture, on the laptop, before it is sent.

By default straight lines come out straight over the frame's full height, in 4:3, which is the
widest straight 4:3 view with no empty edges. The middle shrinks a little to make room. Everything
about this camera is handled here. Everything about the model, like squashing to its square, is
the server's.

The lens is `r = per_radian * (theta + k1 * theta^3)`, with `per_radian = width / source_hfov`.
`source_hfov_deg` of 140° is how many pixels it spends per degree in the middle, measured by
`scripts/check_lens.py --measure`. `k1` then makes the frame's side 80° from the middle, the data
sheet's 160° across. **The edges are an estimate:** nobody has calibrated this lens against a chart,
and the SIYI FPV app has a distortion correction switch that would change the answer.

Passing `view_hfov_deg` and `size` gives a narrower pinhole view instead. The renders were 29.86°
across and 4:3 (24 mm on a 12.8 mm sensor), which `TRAINING_HFOV_DEG` and `VIEW_SIZE` reproduce, and
that is what the pilot sends the server.
"""

from __future__ import annotations

import math

RENDER_FOCAL_MM = 24.0
RENDER_SENSOR_W_MM = 12.8
TRAINING_HFOV_DEG = math.degrees(2.0 * math.atan(RENDER_SENSOR_W_MM / (2.0 * RENDER_FOCAL_MM)))
VIEW_SIZE = (512, 384)      # 4:3 like the renders
A2_MINI_HFOV_DEG = 140.0    # measured on our camera at 1280x720, 2026-11-04: 136 to 141 over 4 runs
A2_MINI_K1 = -0.0644        # 80° at the frame's side, given the above


class LensView:
    def __init__(self, source_hfov_deg: float = A2_MINI_HFOV_DEG, size: tuple | None = None,
                 view_hfov_deg: float | None = None, k1: float = A2_MINI_K1):
        if view_hfov_deg is not None and source_hfov_deg <= view_hfov_deg:
            raise ValueError(f"a {source_hfov_deg}° lens cannot give a {view_hfov_deg}° view")
        self.source_hfov_deg = float(source_hfov_deg)
        self.size = size
        self.view_hfov_deg = view_hfov_deg
        self.k1 = float(k1)
        self._maps = {}

    def angle_at(self, radius_px: float, width: int) -> float:
        """How far from the middle, in radians, the lens sees at this distance in the frame."""
        rho = radius_px * math.radians(self.source_hfov_deg) / width
        theta = rho
        for _ in range(20):
            theta -= (theta + self.k1 * theta ** 3 - rho) / (1.0 + 3.0 * self.k1 * theta ** 2)
        return theta

    def _build(self, height: int, width: int):
        import cv2
        import numpy as np

        w, h = self.size or (round(height * 4 / 3), height)
        per_radian = width / math.radians(self.source_hfov_deg)
        focal_view = (h / 2.0 / math.tan(self.angle_at(height / 2.0, width))
                      if self.view_hfov_deg is None
                      else w / (2.0 * math.tan(math.radians(self.view_hfov_deg) / 2.0)))
        x, y = np.meshgrid(np.arange(w) + 0.5 - w / 2.0, np.arange(h) + 0.5 - h / 2.0)
        radius = np.hypot(x, y)
        theta = np.arctan2(radius, focal_view)
        scale = per_radian * (theta + self.k1 * theta ** 3) / np.maximum(radius, 1e-9)
        return (cv2.convertMaps((x * scale + width / 2.0 - 0.5).astype(np.float32),
                                (y * scale + height / 2.0 - 0.5).astype(np.float32),
                                cv2.CV_16SC2))

    def __call__(self, image):
        import cv2

        shape = image.shape[:2]
        if shape not in self._maps:
            self._maps[shape] = self._build(*shape)
        return cv2.remap(image, *self._maps[shape], cv2.INTER_LINEAR)


def image_shift(before, after) -> tuple[float, float, float]:
    """How far the middle of the picture moved, in pixels: `(dx, dy, confidence)`."""
    import cv2
    import numpy as np

    def middle(image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        side = min(h, w) // 2
        top, left = (h - side) // 2, (w - side) // 2
        return np.float32(grey[top:top + side, left:left + side])

    a, b = middle(before), middle(after)
    (dx, dy), confidence = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return float(dx), float(dy), float(confidence)


def hfov_from_tilt(shift_px: float, tilt_deg: float, width: int) -> float:
    """The `source_hfov_deg` of a lens whose picture moved `shift_px` when tilted `tilt_deg`."""
    return width * abs(tilt_deg) / abs(shift_px)
