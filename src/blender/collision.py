"""Collision detection between placed objects and scene geometry."""

from __future__ import annotations

import bpy
from mathutils import Vector

from src.blender.bbox import get_world_bbox


def check_collisions(imported_objects, scene_meshes, margin=0.05):
    """Check if the imported object's bbox overlaps with scene geometry.

    Args:
        imported_objects: List of imported bpy objects.
        scene_meshes: List of scene mesh objects to check against.
        margin: Shrink the object bbox by this amount to allow touching.

    Returns:
        List of dicts with collision info: [{"object": name, "overlap": float}, ...]
        Empty list if no collisions.
    """
    obj_bbox = get_world_bbox(imported_objects)
    if obj_bbox is None:
        return []

    obj_min, obj_max = obj_bbox
    # Shrink bbox by margin to allow objects resting on surfaces
    obj_min = Vector((obj_min.x + margin, obj_min.y + margin, obj_min.z + margin))
    obj_max = Vector((obj_max.x - margin, obj_max.y - margin, obj_max.z - margin))

    if obj_min.x >= obj_max.x or obj_min.y >= obj_max.y or obj_min.z >= obj_max.z:
        return []

    collisions = []
    for scene_obj in scene_meshes:
        if not hasattr(scene_obj, "bound_box"):
            continue
        # Get scene object's world bbox
        s_min = Vector((float("inf"),) * 3)
        s_max = Vector((float("-inf"),) * 3)
        for corner in scene_obj.bound_box:
            w = scene_obj.matrix_world @ Vector(corner)
            s_min = Vector((min(s_min.x, w.x), min(s_min.y, w.y), min(s_min.z, w.z)))
            s_max = Vector((max(s_max.x, w.x), max(s_max.y, w.y), max(s_max.z, w.z)))

        # AABB overlap test
        if (obj_min.x <= s_max.x and obj_max.x >= s_min.x and
            obj_min.y <= s_max.y and obj_max.y >= s_min.y and
            obj_min.z <= s_max.z and obj_max.z >= s_min.z):

            # Compute overlap volume as a rough measure
            overlap_x = max(0, min(obj_max.x, s_max.x) - max(obj_min.x, s_min.x))
            overlap_y = max(0, min(obj_max.y, s_max.y) - max(obj_min.y, s_min.y))
            overlap_z = max(0, min(obj_max.z, s_max.z) - max(obj_min.z, s_min.z))
            overlap_vol = overlap_x * overlap_y * overlap_z

            obj_vol = ((obj_max.x - obj_min.x) *
                       (obj_max.y - obj_min.y) *
                       (obj_max.z - obj_min.z))
            overlap_ratio = overlap_vol / max(obj_vol, 0.001)

            if overlap_ratio > 0.01:  # > 1% overlap
                collisions.append({
                    "object": scene_obj.name,
                    "overlap_ratio": round(overlap_ratio, 3),
                })

    # Sort by overlap ratio descending
    collisions.sort(key=lambda c: c["overlap_ratio"], reverse=True)
    return collisions[:5]  # top 5 colliding objects
