"""Bounding box utilities for Blender objects."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector


def get_world_bbox(
    objs: list[bpy.types.Object],
) -> tuple[Vector, Vector] | None:
    """Compute the world-space axis-aligned bounding box of a list of objects."""
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in objs:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            min_v = Vector((min(min_v.x, w.x), min(min_v.y, w.y), min(min_v.z, w.z)))
            max_v = Vector((max(max_v.x, w.x), max(max_v.y, w.y), max(max_v.z, w.z)))
            found = True
    return (min_v, max_v) if found else None


def project_bbox_2d(target_objects, resolution):
    """Project 3D bounding boxes of objects to 2D pixel coordinates.

    Returns [x1, y1, x2, y2] or None if not visible.
    """
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return None

    w, h = resolution
    xs, ys = [], []
    for obj in target_objects:
        for corner in obj.bound_box:
            co = world_to_camera_view(scene, cam, obj.matrix_world @ Vector(corner))
            if co.z <= 0:
                continue
            xs.append(co.x * w)
            ys.append((1.0 - co.y) * h)

    if len(xs) < 2:
        return None

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(w), max(xs))
    y2 = min(float(h), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]
