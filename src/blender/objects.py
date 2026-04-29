"""Object import, parenting, scaling, and visibility helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
from mathutils import Vector

from src.assets.object_sizes import expected_object_height_m as _expected_object_height_m
from src.blender.bbox import get_world_bbox


def import_object(obj_file: Path) -> list[bpy.types.Object]:
    """Import a 3D object file into the current scene. Returns new objects."""
    before = {obj.name for obj in bpy.data.objects}
    before_images = {img.name for img in bpy.data.images}
    ext = obj_file.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(obj_file))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(obj_file))
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_file))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(obj_file))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(obj_file), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported format: {obj_file}")
    # Fix textures for newly imported images only
    new_images = {img.name for img in bpy.data.images} - before_images
    _fix_imported_textures(obj_file, new_images)
    return [obj for obj in bpy.data.objects if obj.name not in before]


def _fix_imported_textures(source_file: Path, image_names: set[str]) -> None:
    """Fix texture paths for newly imported images only.

    For packed images with broken file paths, clear the filepath so Blender
    uses the packed data. For unpacked images, resolve paths relative to the
    source file's directory.
    """
    source_dir = source_file.resolve().parent
    for image in bpy.data.images:
        if image.name not in image_names:
            continue
        # Packed images: clear broken path so Blender uses packed data
        if image.packed_file is not None:
            if image.filepath:
                abs_path = bpy.path.abspath(image.filepath)
                if not os.path.isfile(abs_path):
                    image.filepath = ""
            continue
        # Unpacked images: try resolving relative to source file
        if not image.filepath:
            continue
        abs_path = bpy.path.abspath(image.filepath)
        if os.path.isfile(abs_path):
            continue
        raw = image.filepath
        if raw.startswith("//"):
            candidate = source_dir / raw[2:]
        else:
            candidate = source_dir / raw
        if candidate.exists():
            image.filepath = str(candidate.resolve())
            try:
                image.reload()
            except Exception:
                pass


def parent_and_center(imported: list[bpy.types.Object]) -> bpy.types.Object:
    """Parent imported objects under a root empty, centered at the bottom."""
    root = bpy.data.objects.new("PlacedObject", None)
    bpy.context.scene.collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent in imported_set:
            continue
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    bpy.context.view_layer.update()
    bbox = get_world_bbox(imported)
    if bbox:
        mn, mx = bbox
        bottom_center = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
        root.location -= bottom_center
    bpy.context.view_layer.update()
    return root


def auto_fix_orientation(root, imported):
    """Detect and fix wrong up-axis orientation (Y-up vs Z-up).

    Returns the applied correction as (rx, ry, rz) in degrees.
    """
    bbox = get_world_bbox(imported)
    if bbox is None:
        return (0.0, 0.0, 0.0)

    size = bbox[1] - bbox[0]
    height = size.z
    width = max(size.x, size.y)

    if height < 0.01:
        return (0.0, 0.0, 0.0)

    aspect = width / height

    if aspect > 2.0:
        for deg in (-90, 90):
            root.rotation_euler = (math.radians(deg), 0, 0)
            bpy.context.view_layer.update()

            bbox_new = get_world_bbox(imported)
            if bbox_new:
                new_size = bbox_new[1] - bbox_new[0]
                new_aspect = max(new_size.x, new_size.y) / max(new_size.z, 0.01)
                if new_aspect < aspect:
                    mn, mx = bbox_new
                    bottom_center = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
                    root.location -= bottom_center
                    bpy.context.view_layer.update()
                    print(f"Auto-rotated {deg}deg X (aspect {aspect:.1f} → {new_aspect:.1f})")
                    return (float(deg), 0.0, 0.0)

        root.rotation_euler = (0, 0, 0)
        bpy.context.view_layer.update()

    return (0.0, 0.0, 0.0)


def normalize_to_metric(imported_root, imported_objs, scene_meshes,
                        object_path, exclude_names=None) -> float:
    """Ensure both object and scene are at correct absolute metric scale.

    Step 1: Scale the OBJECT to its expected real-world metric size.
            A wooden cat should be ~0.35m, a person ~1.7m, etc.
    Step 2: Check if the SCENE is at a reasonable metric scale.
            If it's way too large (>60m for an interior), shrink it.

    Returns the total correction factor applied.
    """
    expected_h, label = _expected_object_height_m(object_path)
    obj_bbox = get_world_bbox(imported_objs)
    if obj_bbox is None:
        return 1.0
    actual_h = obj_bbox[1].z - obj_bbox[0].z
    if actual_h < 0.001:
        return 1.0

    exclude = set(exclude_names or set())

    # Step 1: Scale OBJECT to correct absolute metric size — but trust the
    # imported size if it's already in the right ballpark. Keyword-based
    # category guesses are fragile (a "Buddy-Sitting" person has bbox ≈ 1.0m,
    # not 1.7m; "Photographer-on-a-bar-stool" is sitting tall ≈ 1.3m).
    # Only override when the imported size is wildly off (e.g. 10×+ wrong
    # due to unit confusion: cm vs m, or mm vs m).
    obj_correction = expected_h / actual_h
    # Wide plausible band: allow 0.3×–3.5× of the keyword-guessed height.
    # Visual scan of our 102 assets shows real diversity:
    #   - sitting human ~0.85m vs standing 1.7m  (0.5× ratio)
    #   - mini decorative snowman ~0.5m vs large 1.5m  (3× ratio)
    #   - tall human in cape ~2m vs default 1.7m  (1.18× ratio)
    # Outside 0.3-3.5× we flag as a unit error (e.g. cm/mm/m confusion,
    # like rp_posedplus arriving at 169m).
    if 0.3 <= obj_correction <= 3.5:
        print(f"Object OK: {label} is {actual_h:.2f}m "
              f"(expected ~{expected_h:.2f}m, ratio {obj_correction:.2f}x)")
        obj_correction = 1.0
    else:
        print(f"Scaling OBJECT: {label} {actual_h:.2f}m → {expected_h:.2f}m "
              f"(factor {obj_correction:.3f}; way outside plausible range — "
              f"likely unit error)")
        imported_root.scale = tuple(s * obj_correction for s in imported_root.scale)
        imported_root.location = tuple(v * obj_correction for v in imported_root.location)
        bpy.context.view_layer.update()

    # Step 2: Scene scale — don't auto-adjust.
    # The object is now at correct absolute metric size.
    # The scene may be in different units but we can't reliably detect that
    # without visual inspection. The VLM will flag scale mismatches via
    # its scale_factor feedback during placement evaluation.

    return obj_correction


# Known furniture dimensions in metric (height in meters)
_FURNITURE_REFERENCES = [
    # (keywords, expected_height, tolerance_factor)
    (("door", "doorframe", "door_frame"), 2.1, 1.5),
    (("chair",), 0.85, 1.5),
    (("table",), 0.75, 1.5),
    (("counter", "countertop"), 0.90, 1.5),
    (("sink",), 0.85, 1.5),
    (("cabinet",), 0.90, 2.0),
    (("window",), 1.2, 2.0),
    (("bed",), 0.55, 1.5),
    (("sofa", "couch"), 0.85, 1.5),
    (("stool",), 0.75, 1.5),
]


def _estimate_scene_scale_from_furniture(scene_meshes) -> float:
    """Estimate scene scale by comparing furniture sizes to known metric values.

    Scans scene objects for names matching common furniture (door, chair, table).
    Compares their height to expected metric size. Returns the median correction
    factor, or 1.0 if no furniture is found.
    """
    from mathutils import Vector

    ratios = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        for keywords, expected_h, tolerance in _FURNITURE_REFERENCES:
            if any(kw in name for kw in keywords):
                # Compute object height
                bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
                h = max(v.z for v in bb) - min(v.z for v in bb)
                if h < 0.01:
                    continue
                ratio = expected_h / h
                # Only count if the ratio suggests a real unit mismatch
                # (within tolerance = might just be a small/large version)
                if ratio < 1.0 / tolerance or ratio > tolerance:
                    ratios.append(ratio)
                    print(f"  Furniture ref: {obj.name} is {h:.2f}m, "
                          f"expected ~{expected_h:.2f}m (ratio {ratio:.2f})")
                break

    if not ratios:
        print("No furniture references found — assuming scene is metric")
        return 1.0

    # Use median ratio
    ratios.sort()
    median = ratios[len(ratios) // 2]

    # Only apply if the correction is significant
    if 0.8 < median < 1.25:
        print(f"Furniture check: median ratio {median:.2f} — scene is metric")
        return 1.0

    print(f"Furniture check: {len(ratios)} refs, median ratio {median:.2f} "
          f"→ scaling scene by {median:.4f}")
    return median


# Backward compatibility shim - old name still used in some places
def normalize_scene_scale(scene_meshes, exclude_names=None) -> float:
    """Deprecated: kept for backward compatibility. No-op now.

    Use normalize_to_metric() with the imported object instead.
    """
    return 1.0


def auto_scale(root, imported, scene_meshes, override_scale=0.0):
    """Scale object to reasonable size relative to scene.

    After normalize_scene_scale() has been called, the scene is in metric
    units and objects at real-world scale should fit. This handles edge
    cases where the object itself is in wrong units.

    Returns the applied scale factor (1.0 if no scaling).
    """
    if override_scale > 0:
        root.scale = (override_scale, override_scale, override_scale)
        bpy.context.view_layer.update()
        print(f"Override scale: {override_scale:.4f}")
        return override_scale

    bbox = get_world_bbox(imported)
    if bbox is None:
        return 1.0
    obj_size = bbox[1] - bbox[0]
    obj_height = obj_size.z

    def _apply(scale, reason):
        root.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        print(f"Auto-scaled object by {scale:.4f} ({reason})")
        return scale

    # Object in cm/mm (height >> 10 units for something that should be ~1-2m)
    if obj_height > 10:
        return _apply(1.7 / obj_height, "object likely in cm/mm")
    # Object unreasonably tiny
    if obj_height < 0.01:
        return _apply(1.7 / obj_height, "object too tiny")

    print(f"Object scale OK (height={obj_height:.2f}m)")
    return 1.0


def set_hidden(objs, hidden):
    """Hide/unhide objects. Returns previous state for restore_hidden()."""
    prev = []
    for obj in objs:
        prev.append((obj, bool(obj.hide_viewport), bool(obj.hide_render)))
        try:
            obj.hide_set(hidden)
        except Exception:
            obj.hide_viewport = hidden
        obj.hide_render = hidden
    return prev


def restore_hidden(prev):
    """Restore visibility state saved by set_hidden()."""
    for obj, hv, hr in prev:
        try:
            obj.hide_set(hv)
        except Exception:
            obj.hide_viewport = hv
        obj.hide_render = hr


def fix_missing_textures():
    """Replace truly missing texture images with a neutral gray fallback.

    Skips images that have packed data — those render correctly even if
    the external filepath is broken (common in downloaded .blend files
    with paths to the original artist's machine).
    """
    replaced = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                # Skip packed images — they render fine regardless of filepath
                if node.image.packed_file is not None:
                    continue
                fpath = bpy.path.abspath(node.image.filepath)
                if fpath and not os.path.isfile(fpath):
                    color_node = mat.node_tree.nodes.new("ShaderNodeRGB")
                    color_node.outputs[0].default_value = (0.6, 0.6, 0.6, 1.0)
                    for link in mat.node_tree.links:
                        if link.from_node == node:
                            mat.node_tree.links.new(
                                color_node.outputs[0], link.to_socket
                            )
                    replaced += 1
    if replaced:
        print(f"Replaced {replaced} truly missing textures with gray fallback")
