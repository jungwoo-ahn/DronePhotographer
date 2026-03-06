from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union

from drones.dronecore.camera import build_camera
from drones.dronecore.drone import DroneEnvironment, Pose
from src.scenes.scene import ensure_camera, set_render_settings


class BlenderDroneEnvironment(DroneEnvironment):
    def __init__(
        self,
        scene,
        camera_info=None,
        camera_name: str = "DroneCamera",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.scene = scene
        self.camera_info = build_camera(camera_info)
        self.camera = ensure_camera(scene, camera_name)
        self.output_dir = Path(output_dir) if output_dir else None
        self._frame_index = 0

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._configure_camera()
        self._configure_render()

    def _configure_camera(self) -> None:
        if not isinstance(self.camera_info, dict):
            return
        cam = self.camera.data
        if self.camera_info.get("focal_length") is not None:
            cam.lens = float(self.camera_info["focal_length"])
        if self.camera_info.get("sensor_width") is not None:
            cam.sensor_width = float(self.camera_info["sensor_width"])
        if self.camera_info.get("sensor_height") is not None:
            cam.sensor_height = float(self.camera_info["sensor_height"])
        if self.camera_info.get("clip_start") is not None:
            cam.clip_start = float(self.camera_info["clip_start"])
        if self.camera_info.get("clip_end") is not None:
            cam.clip_end = float(self.camera_info["clip_end"])

    def _configure_render(self) -> None:
        if not isinstance(self.camera_info, dict):
            return
        set_render_settings(
            self.scene,
            engine=self.camera_info.get("render_engine", "CYCLES"),
            resolution=self.camera_info.get("resolution"),
            image_format=self.camera_info.get("image_format", "PNG"),
        )

    def _next_output_path(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        self._frame_index += 1
        image_format = "png"
        if isinstance(self.camera_info, dict):
            image_format = str(self.camera_info.get("image_format", "PNG")).lower()
        return self.output_dir / f"frame_{self._frame_index:05d}.{image_format}"

    def move(self, delta_position, delta_orientation) -> Pose:
        self.camera.location = tuple(
            value + delta for value, delta in zip(self.camera.location, delta_position)
        )
        self.camera.rotation_euler = tuple(
            value + delta for value, delta in zip(self.camera.rotation_euler, delta_orientation)
        )
        return self.get_pose()

    def move_to(self, position, orientation) -> Pose:
        self.camera.location = tuple(position)
        self.camera.rotation_euler = tuple(orientation)
        return self.get_pose()

    def get_pose(self) -> Pose:
        position = tuple(self.camera.location)
        orientation = tuple(self.camera.rotation_euler)
        return position, orientation

    def render(self) -> Any:
        import bpy

        output_path = self._next_output_path()
        if output_path is not None:
            self.scene.render.filepath = str(output_path)
            bpy.ops.render.render(write_still=True)
            return str(output_path)
        bpy.ops.render.render()
        return None
