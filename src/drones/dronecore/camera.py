from __future__ import annotations
from typing import Any, Dict

DEFAULT_CAMERA_INFO: Dict[str, Any] = {
    "resolution": (1024, 768),
    "focal_length": 50.0,
    "sensor_width": 36.0,
    "sensor_height": 24.0,
    "image_format": "PNG",
    "render_engine": "CYCLES",
}


def build_camera(camera_info: Any = None) -> Any:
    if camera_info is None:
        return dict(DEFAULT_CAMERA_INFO)
    if isinstance(camera_info, dict):
        merged = dict(DEFAULT_CAMERA_INFO)
        merged.update(camera_info)
        return merged
    return camera_info
