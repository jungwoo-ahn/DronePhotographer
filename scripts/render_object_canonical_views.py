"""Run inside Blender. Renders 4 canonical views (FRONT/RIGHT/BACK/LEFT) of
a single object placed at origin with rotation_z=0, on a neutral ground
plane, with a red 3D arrow lying on the ground pointing in the renderer's
*assumed* object_forward direction (+Y world for rot_z=0).

If the FRONT render shows the character's face, the +Y assumption is
correct. If FRONT shows the back of the head, that asset needs
``rotation_z_deg=180``.

Run via:

    blender --background --python scripts/render_object_canonical_views.py -- \
        --object_file <path.blend> --output_dir <dir> [--resolution 512] [--samples 16]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.blender.objects import (
    import_object, parent_and_center, auto_fix_orientation,
    fix_missing_textures,
)
from src.blender.bbox import get_world_bbox


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--object_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                   help="BLENDER_EEVEE_NEXT or CYCLES")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.45, 0.5, 0.55, 1.0)
        bg.inputs[1].default_value = 0.6
    return scene


def add_lighting():
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(50), math.radians(15), math.radians(30))

    bpy.ops.object.light_add(type="SUN", location=(-3, 4, 6))
    fill = bpy.context.object
    fill.data.energy = 1.5
    fill.rotation_euler = (math.radians(120), 0, math.radians(180))


def add_ground():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "Ground"
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.58, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.95
    plane.data.materials.append(mat)


def make_arrow_material():
    mat = bpy.data.materials.new("ArrowMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.95, 0.10, 0.10, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        # Blender 4.x renamed "Emission" → "Emission Color" / "Emission Strength"
        for col_key in ("Emission Color", "Emission"):
            if col_key in bsdf.inputs:
                bsdf.inputs[col_key].default_value = (1.0, 0.05, 0.05, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.5
    return mat


def add_ground_arrow(obj_xy_center, ground_z, length, mat):
    """Build a 3D ground arrow as cylinder shaft + cone head, lying flat at
    z=ground_z, pointing along world +Y (the renderer's assumed front).
    Origin sits at the object's XY center."""
    head_frac = 0.35
    shaft_len = length * (1 - head_frac)
    head_len = length * head_frac
    shaft_r = max(0.03, length * 0.05)
    head_r = max(0.06, length * 0.13)

    # Shaft — cylinder oriented along +Y (rotate 90° about X).
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_r, depth=shaft_len,
        location=(0, shaft_len / 2, ground_z + shaft_r),
        rotation=(math.radians(90), 0, 0),
    )
    shaft = bpy.context.object
    shaft.name = "ArrowShaft"
    shaft.data.materials.append(mat)

    # Head — cone oriented along +Y.
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_r, radius2=0,
        depth=head_len,
        location=(0, shaft_len + head_len / 2, ground_z + shaft_r),
        rotation=(math.radians(-90), 0, 0),
    )
    head = bpy.context.object
    head.name = "ArrowHead"
    head.data.materials.append(mat)

    # Group under empty centred at object XY for clean translation.
    bpy.ops.object.empty_add(type="PLAIN_AXES",
                             location=(obj_xy_center[0], obj_xy_center[1], ground_z))
    root = bpy.context.object
    root.name = "FrontArrow"
    shaft.parent = root
    head.parent = root
    bpy.context.view_layer.update()
    return root


def setup_camera(target, view_dir, dist, height_offset):
    """Place camera at target + view_dir*dist + (0,0,height_offset), looking at target."""
    target = Vector(target)
    view_dir = Vector(view_dir).normalized()
    cam_pos = target + view_dir * dist + Vector((0, 0, height_offset))

    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = cam_pos

    direction = (target - cam_pos).normalized()
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    cam.data.lens = 50
    cam.data.sensor_width = 36
    return cam


def configure_render(scene, engine, resolution, samples):
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    if engine == "BLENDER_EEVEE_NEXT" and hasattr(scene, "eevee"):
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass
    elif engine == "CYCLES":
        scene.cycles.samples = samples


def render_to(out_path):
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = reset_scene()
    add_lighting()
    add_ground()
    arrow_mat = make_arrow_material()

    print(f"Importing {args.object_file}")
    imported = import_object(Path(args.object_file))
    if not imported:
        raise RuntimeError("import_object returned no objects")
    root = parent_and_center(imported)
    auto_fix_orientation(root, imported)
    fix_missing_textures()
    bpy.context.view_layer.update()

    bbox = get_world_bbox(imported)
    if bbox is None:
        raise RuntimeError("could not compute world bbox")
    mn, mx = bbox
    dims = mx - mn
    obj_height = float(dims.z)
    obj_xy = float(max(dims.x, dims.y))
    obj_center = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, (mn.z + mx.z) / 2))

    arrow_len = max(0.8, obj_xy * 1.0)
    add_ground_arrow(
        obj_xy_center=(obj_center.x, obj_center.y),
        ground_z=mn.z + 0.005,
        length=arrow_len,
        mat=arrow_mat,
    )

    cam_dist = max(2.5, obj_height * 2.2)
    eye_target = Vector((obj_center.x, obj_center.y, mn.z + obj_height * 0.55))

    views = {
        "FRONT": Vector((0, 1, 0)),
        "RIGHT": Vector((1, 0, 0)),
        "BACK":  Vector((0, -1, 0)),
        "LEFT":  Vector((-1, 0, 0)),
    }

    configure_render(scene, args.engine, args.resolution, args.samples)

    rendered = {}
    for label, view_dir in views.items():
        setup_camera(eye_target, view_dir, cam_dist, height_offset=0.0)
        out_path = out_dir / f"{label.lower()}.png"
        print(f"  rendering {label} -> {out_path.name}")
        render_to(out_path)
        rendered[label] = out_path.name

    meta = {
        "object_file": args.object_file,
        "object_dims": [float(dims.x), float(dims.y), float(dims.z)],
        "obj_height": obj_height,
        "obj_xy": obj_xy,
        "arrow_len": arrow_len,
        "cam_dist": cam_dist,
        "rendered": rendered,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done — {len(rendered)} views in {out_dir}")


if __name__ == "__main__":
    main()
