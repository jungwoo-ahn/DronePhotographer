from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def open_scene(scene_file: str):
	"""Open a .blend file and return bpy.context.scene.

	Must be executed inside Blender (requires `bpy`).
	"""

	import bpy

	bpy.ops.wm.open_mainfile(filepath=str(scene_file))
	return bpy.context.scene


def set_nishita_sky(strength: float) -> None:
	"""Set Nishita sky as world background if strength > 0."""

	if strength <= 0:
		return

	import bpy

	scene = bpy.context.scene
	world = scene.world or bpy.data.worlds.new("World")
	scene.world = world
	world.use_nodes = True

	nodes = world.node_tree.nodes
	links = world.node_tree.links
	nodes.clear()

	bg = nodes.new(type="ShaderNodeBackground")
	sky = nodes.new(type="ShaderNodeTexSky")
	sky.sky_type = "NISHITA"
	bg.inputs["Strength"].default_value = float(strength)
	output = nodes.new(type="ShaderNodeOutputWorld")

	links.new(sky.outputs["Color"], bg.inputs["Color"])
	links.new(bg.outputs["Background"], output.inputs["Surface"])


def ensure_camera(scene, camera_name: str = "RenderCamera"):
	"""Ensure scene has an active camera object and return it."""

	import bpy

	cam = scene.camera
	if cam is None:
		for obj in bpy.data.objects:
			if obj.type == "CAMERA":
				cam = obj
				break
	if cam is None:
		cam_data = bpy.data.cameras.new(camera_name)
		cam = bpy.data.objects.new(camera_name, cam_data)
		scene.collection.objects.link(cam)
	scene.camera = cam
	return cam


def set_render_settings(
	scene,
	*,
	engine: str = "CYCLES",
	resolution: Optional[Tuple[int, int]] = None,
	image_format: str = "PNG",
) -> None:
	"""Apply basic render settings."""

	scene.render.engine = str(engine)
	if resolution is not None:
		scene.render.resolution_x = int(resolution[0])
		scene.render.resolution_y = int(resolution[1])
		scene.render.resolution_percentage = 100
	scene.render.image_settings.file_format = str(image_format)
