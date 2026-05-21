from __future__ import annotations

import json
import math
import random
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from src.scenes.scene import open_scene, set_nishita_sky
from src.utils.rotation_utils import apply_camera_local_action, orthonormalize_forward_up
from render_object import place_imported_object


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    camera = scene.camera
    if camera is None:
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                camera = obj
                break
    if camera is None:
        cam_data = bpy.data.cameras.new("RenderCamera")
        camera = bpy.data.objects.new("RenderCamera", cam_data)
        scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def _prepare_camera(camera: bpy.types.Object) -> None:
    if camera.parent is not None:
        camera.parent = None
    if camera.constraints:
        for constraint in list(camera.constraints):
            camera.constraints.remove(constraint)
    if camera.animation_data is not None:
        camera.animation_data_clear()


def _configure_gpu(
    scene: bpy.types.Scene,
    *,
    disable_gpu: bool,
    gpu_backend: str,
    gpu_devices: list[int] | None,
) -> None:
    scene.render.engine = "CYCLES"
    if disable_gpu:
        scene.cycles.device = "CPU"
        return

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        scene.cycles.device = "CPU"
        return

    prefs = cycles_addon.preferences
    backend_order = ("OPTIX", "CUDA") if gpu_backend == "AUTO" else (gpu_backend,)
    for backend in backend_order:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue

        backend_devices = [device for device in prefs.devices if device.type == backend]
        if not backend_devices:
            continue

        selected = set()
        if gpu_devices:
            selected = {index for index in gpu_devices if 0 <= index < len(backend_devices)}

        for device in prefs.devices:
            device.use = False
        for index, device in enumerate(backend_devices):
            device.use = index in selected if selected else True

        scene.cycles.device = "GPU"
        return

    scene.cycles.device = "CPU"


def _set_render_settings(scene: bpy.types.Scene, options: dict[str, object]) -> None:
    resolution = options.get("resolution", [1024, 768])
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_persistent_data = bool(options.get("persistent_data", False))
    scene.render.film_transparent = False

    cycles = scene.cycles
    cycles.samples = int(options.get("samples", 64))
    cycles.max_bounces = int(options.get("max_bounces", 4))
    cycles.diffuse_bounces = int(options.get("diffuse_bounces", 2))
    cycles.glossy_bounces = int(options.get("glossy_bounces", 2))
    cycles.transmission_bounces = int(options.get("transmission_bounces", 2))
    cycles.volume_bounces = int(options.get("volume_bounces", 0))
    cycles.transparent_max_bounces = int(options.get("transparent_max_bounces", 4))
    if hasattr(cycles, "use_adaptive_sampling"):
        cycles.use_adaptive_sampling = bool(options.get("adaptive_sampling", False))
    if hasattr(cycles, "adaptive_threshold"):
        cycles.adaptive_threshold = float(options.get("adaptive_threshold", 0.01))

    blender_threads = int(options.get("blender_threads", 4))
    if hasattr(scene.render, "threads_mode"):
        scene.render.threads_mode = "FIXED"
        scene.render.threads = max(1, blender_threads)


def _vector_to_list(vector: Vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _rotation_matrix_from_forward_up(forward: np.ndarray, up: np.ndarray) -> Matrix:
    forward_norm, up_norm = orthonormalize_forward_up(forward, up)
    forward_vec = Vector(forward_norm.tolist())
    up_vec = Vector(up_norm.tolist())
    right_vec = forward_vec.cross(up_vec)
    right_vec.normalize()
    up_vec = right_vec.cross(forward_vec)
    up_vec.normalize()
    return Matrix(
        (
            (right_vec.x, up_vec.x, -forward_vec.x),
            (right_vec.y, up_vec.y, -forward_vec.y),
            (right_vec.z, up_vec.z, -forward_vec.z),
        )
    )


def _random_direction(rng: random.Random, hemisphere: bool) -> Vector:
    u = rng.random()
    v = rng.random()
    theta = 2.0 * math.pi * u
    z = v if hemisphere else (2.0 * v - 1.0)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return Vector((radius * math.cos(theta), radius * math.sin(theta), z))


def _look_at_matrix(camera_position: Vector, target: Vector) -> Matrix:
    world_up = Vector((0.0, 0.0, 1.0))
    forward = (target - camera_position).normalized()
    right = forward.cross(world_up)
    if right.length < 1e-6:
        right = forward.cross(Vector((0.0, 1.0, 0.0)))
    right.normalize()
    up = right.cross(forward).normalized()
    return Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )


class BlenderDrone:
    """Simple Blender camera controller reconstructed from a render run."""

    def __init__(self, run_info: dict[str, object], run_info_path: str | Path) -> None:
        self.run_info = dict(run_info)
        self.run_info_path = _to_path(run_info_path)
        self.options = dict(self.run_info.get("options", {}))
        self.scene_path = Path(str(self.run_info["input_scene"]))
        self.scene = open_scene(str(self.scene_path))
        self.camera = _ensure_camera(self.scene)
        _prepare_camera(self.camera)

        # Re-import external object if needed
        input_object = self.run_info.get("input_object")
        if input_object is not None:
            obj_pos_raw = self.options.get("object_position", [0, 0, 0])
            scale = float(self.run_info.get("scale", 1.0))
            rotation_z_deg = float(self.run_info.get("rotation_z_deg", 0.0))
            place_imported_object(
                self.scene, input_object, obj_pos_raw,
                rotation_z_deg=rotation_z_deg, scale=scale,
            )

        self.object_position = np.asarray(self.options["object_position"], dtype=np.float32)
        self.camera_radius_range = [float(value) for value in self.options.get("camera_radius_range", [2.0, 10.0])]
        self.camera_direction_offsets = [
            float(value) for value in self.options.get("camera_direction_offsets", [30.0, 30.0, 30.0])
        ]
        self.hemisphere = bool(self.options.get("hemisphere", False))

        self.camera.data.lens = float(self.options.get("focal_length", 24.0))
        self.camera.data.sensor_width = float(self.options.get("sensor_width", 12.8))
        self.camera.data.sensor_height = float(self.options.get("sensor_height", 9.6))
        if float(self.options.get("sky_strength", 0.0)) > 0.0:
            set_nishita_sky(float(self.options["sky_strength"]))

        _configure_gpu(
            self.scene,
            disable_gpu=bool(self.options.get("render_device", "CPU") == "CPU"),
            gpu_backend=str(self.options.get("gpu_backend", "AUTO")),
            gpu_devices=self.options.get("gpu_devices"),
        )
        _set_render_settings(self.scene, self.options)
        self._detector = None

    @classmethod
    def from_run_info(cls, run_info_path: str | Path) -> "BlenderDrone":
        run_info_path = _to_path(run_info_path)
        with run_info_path.open("r", encoding="utf-8") as f:
            run_info = json.load(f)
        return cls(run_info=run_info, run_info_path=run_info_path)

    def get_pose(self) -> dict[str, object]:
        rotation = self.camera.matrix_world.to_3x3()
        position = self.camera.matrix_world.translation.copy()
        forward = (rotation @ Vector((0.0, 0.0, -1.0))).normalized()
        up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
        radius = float((position - Vector(self.object_position.tolist())).length)
        return {
            "position": _vector_to_list(position),
            "forward": _vector_to_list(forward),
            "up": _vector_to_list(up),
            "radius": radius,
        }

    def set_pose(
        self,
        position: list[float] | tuple[float, float, float] | np.ndarray,
        forward: list[float] | tuple[float, float, float] | np.ndarray,
        up: list[float] | tuple[float, float, float] | np.ndarray,
    ) -> dict[str, object]:
        position_np = np.asarray(position, dtype=np.float32)
        rotation_matrix = _rotation_matrix_from_forward_up(
            np.asarray(forward, dtype=np.float32),
            np.asarray(up, dtype=np.float32),
        )
        self.camera.matrix_world = Matrix.Translation(Vector(position_np.tolist())) @ rotation_matrix.to_4x4()
        bpy.context.view_layer.update()
        return self.get_pose()

    def sample_initial_pose(self, seed: int | None = None) -> dict[str, object]:
        rng = random.Random(seed)
        object_position = Vector(self.object_position.tolist())
        radius_min, radius_max = sorted(self.camera_radius_range)
        direction = _random_direction(rng, self.hemisphere)
        radius = rng.uniform(radius_min, radius_max)
        camera_position = object_position + direction * radius

        base_rotation = _look_at_matrix(camera_position, object_position)
        yaw_range, pitch_range, roll_range = self.camera_direction_offsets
        yaw_deg = rng.uniform(-yaw_range, yaw_range)
        pitch_deg = rng.uniform(-pitch_range, pitch_range)
        roll_deg = rng.uniform(-roll_range, roll_range)
        offset_rotation = Euler(
            (math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)),
            "XYZ",
        ).to_matrix()
        final_rotation = base_rotation @ offset_rotation
        forward = (final_rotation @ Vector((0.0, 0.0, -1.0))).normalized()
        up = (final_rotation @ Vector((0.0, 1.0, 0.0))).normalized()
        pose = self.set_pose(camera_position, forward, up)
        pose["offsets_deg"] = {
            "yaw": float(yaw_deg),
            "pitch": float(pitch_deg),
            "roll": float(roll_deg),
        }
        return pose

    def apply_local_action(
        self,
        delta_position_local: list[float] | tuple[float, float, float] | np.ndarray,
        delta_rotation_local: list[float] | tuple[float, float, float] | np.ndarray,
    ) -> dict[str, object]:
        pose = self.get_pose()
        next_position, next_forward, next_up = apply_camera_local_action(
            position=np.asarray(pose["position"], dtype=np.float32),
            forward=np.asarray(pose["forward"], dtype=np.float32),
            up=np.asarray(pose["up"], dtype=np.float32),
            delta_position_local=np.asarray(delta_position_local, dtype=np.float32),
            delta_rotation_local=np.asarray(delta_rotation_local, dtype=np.float32),
        )
        return self.set_pose(next_position, next_forward, next_up)

    def render_rgb(self, output_path: str | Path) -> Path:
        output_path = _to_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = output_path.with_name(f"{output_path.stem}.rendering{output_path.suffix}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        self.scene.render.filepath = str(temp_output_path)
        bpy.ops.render.render(write_still=True)
        if not temp_output_path.exists():
            raise FileNotFoundError(f"Blender did not produce render output: {temp_output_path}")
        temp_output_path.replace(output_path)
        return output_path

    def detect_subject(
        self,
        image_path: str | Path,
        caption: str,
        *,
        model_config_path: str | Path = "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path: str | Path = "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[dict[str, object]]:
        if self._detector is None:
            from src.detectors.detector import GroundingDINODetector

            self._detector = GroundingDINODetector(
                model_config_path=model_config_path,
                model_checkpoint_path=model_checkpoint_path,
                device=device,
            )
        detections = self._detector.detect_file(
            image_path=image_path,
            caption=caption,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )
        return [d.as_dict() for d in detections]
