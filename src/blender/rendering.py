"""Render helpers for Blender."""

from __future__ import annotations

import bpy


def render_image(output_path, engine="EEVEE", resolution=(512, 384), samples=16):
    """Configure render settings and render a single image."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    scene.render.film_transparent = False

    if engine == "EEVEE":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = samples
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

    bpy.ops.render.render(write_still=True)
