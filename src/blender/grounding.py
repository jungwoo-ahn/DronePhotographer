"""Programmatic verification that an object is properly grounded on a surface."""

from __future__ import annotations

import bpy
from mathutils import Vector

from src.blender.bbox import get_world_bbox


def verify_grounded(imported, scene_meshes, max_gap: float = 0.05,
                    embed_tolerance: float = 0.05) -> dict:
    """Check whether the imported object is sitting on a surface.

    Casts rays downward from sample points on the bottom of the object's
    bounding box. Returns a dict with diagnostics.

    Args:
        imported: list of imported bpy objects
        scene_meshes: list of scene mesh objects (for ignoring own hits)
        max_gap: max distance from object bottom to surface to count as grounded
        embed_tolerance: how far the object can sink into a surface before counted as embedded

    Returns dict with:
        grounded: bool — at least one ray hits within max_gap
        embedded: bool — surface is above the object's bottom
        gap_min: float — closest gap (positive = above surface, negative = embedded)
        ground_z: float — average Z of detected ground hits
    """
    bbox = get_world_bbox(imported)
    if bbox is None:
        return {"grounded": False, "embedded": False, "gap_min": float("inf"), "ground_z": None}

    bb_min, bb_max = bbox
    bottom_z = bb_min.z
    cx = (bb_min.x + bb_max.x) / 2
    cy = (bb_min.y + bb_max.y) / 2
    half_x = (bb_max.x - bb_min.x) / 2
    half_y = (bb_max.y - bb_min.y) / 2

    # Sample points: 5 — center plus 4 corners (slightly inset)
    inset = 0.7
    sample_xy = [
        (cx, cy),
        (cx - half_x * inset, cy - half_y * inset),
        (cx + half_x * inset, cy - half_y * inset),
        (cx - half_x * inset, cy + half_y * inset),
        (cx + half_x * inset, cy + half_y * inset),
    ]

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    imported_names = {o.name for o in imported}

    # Raycast from above the object's bottom to detect ground
    cast_start_z = bottom_z + 0.5  # start above the bottom

    gaps = []
    ground_zs = []
    for x, y in sample_xy:
        origin = Vector((x, y, cast_start_z))
        hit, loc, normal, _, obj, _ = scene.ray_cast(
            depsgraph, origin, Vector((0, 0, -1))
        )
        if not hit or obj is None:
            continue
        if obj.name in imported_names:
            continue
        # Skip surfaces with steep slope (walls). Accept both face directions
        # since some imported scenes have inverted normals on floors.
        if abs(normal.z) < 0.5:
            continue
        gap = bottom_z - loc.z  # positive = floating above, negative = embedded
        gaps.append(gap)
        ground_zs.append(loc.z)

    if not gaps:
        return {"grounded": False, "embedded": False, "gap_min": float("inf"), "ground_z": None}

    gap_min = min(gaps, key=abs)
    ground_z = sum(ground_zs) / len(ground_zs)
    grounded = abs(gap_min) <= max_gap
    embedded = gap_min < -embed_tolerance

    return {
        "grounded": grounded,
        "embedded": embedded,
        "gap_min": gap_min,
        "ground_z": ground_z,
    }


def snap_to_floor(root, imported, scene_meshes, max_search_dist: float = 1.0) -> bool:
    """Move the root so the object's bottom rests on the nearest floor below.

    Returns True if a floor was found and the object was snapped, False otherwise.
    """
    info = verify_grounded(imported, scene_meshes, max_gap=max_search_dist,
                           embed_tolerance=max_search_dist)
    if info["ground_z"] is None:
        return False

    bbox = get_world_bbox(imported)
    if bbox is None:
        return False

    bottom_z = bbox[0].z
    delta = info["ground_z"] - bottom_z
    if abs(delta) > max_search_dist:
        return False

    root.location.z += delta + 0.005  # tiny offset to avoid Z-fighting
    bpy.context.view_layer.update()
    print(f"snap_to_floor: shifted by {delta:.3f}m to land on z={info['ground_z']:.3f}")
    return True
