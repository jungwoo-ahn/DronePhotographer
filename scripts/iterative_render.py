"""Single-frame Blender render with explicit camera position & look-at target.

Usage (called from bash):
  blender -b -P scripts/iterative_render.py -- \
    --scene_path  "path/to/scene.blend" \
    --object_path "path/to/object.blend" \
    --obj_pos 0.95 8.4 0.01 \
    --obj_rot -160.5 \
    --obj_scale 0.01004 \
    --cam_pos 3.0 10.0 3.0 \
    --cam_target 1.0 8.4 0.8 \
    --output "outputs/iterative_camera_seaside/iter_00.png" \
    --resolution 512 384 \
    --samples 16
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--scene_path", required=True)
    p.add_argument("--object_path", required=True)
    p.add_argument("--obj_pos", nargs=3, type=float, required=True)
    p.add_argument("--obj_rot", type=float, default=0.0)
    p.add_argument("--obj_scale", type=float, default=1.0)
    p.add_argument("--cam_pos", nargs=3, type=float, required=True)
    p.add_argument("--cam_target", nargs=3, type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resolution", nargs=2, type=int, default=[512, 384])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--focal_length", type=float, default=24.0)
    p.add_argument("--sensor_width", type=float, default=12.8)
    p.add_argument("--gpu_device", type=int, default=5)
    split = argv.index("--") + 1 if "--" in argv else len(argv)
    return p.parse_args(argv[split:])


def look_at_matrix(cam_pos, target):
    forward = (target - cam_pos).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up)
    if right.length < 1e-6:
        right = forward.cross(Vector((0.0, 1.0, 0.0)))
    right.normalize()
    up = right.cross(forward).normalized()
    rot = Matrix((
        (right.x, up.x, -forward.x),
        (right.y, up.y, -forward.y),
        (right.z, up.z, -forward.z),
    ))
    return rot


def import_and_place_object(scene, obj_path, position, rot_deg, scale):
    obj_file = Path(obj_path)
    before = {o.name for o in bpy.data.objects}
    with bpy.data.libraries.load(str(obj_file), link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]
    imported = []
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
            imported.append(obj)
    # Parent under a root
    root = bpy.data.objects.new("PlacedObject", None)
    scene.collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent in imported_set:
            continue
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    # Move bottom-center to origin
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    for obj in imported:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            w = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, w.x)
            min_v.y = min(min_v.y, w.y)
            min_v.z = min(min_v.z, w.z)
            max_v.x = max(max_v.x, w.x)
            max_v.y = max(max_v.y, w.y)
            max_v.z = max(max_v.z, w.z)
    bottom_center = Vector(((min_v.x + max_v.x) * 0.5, (min_v.y + max_v.y) * 0.5, min_v.z))
    root.location = root.location - bottom_center
    bpy.context.view_layer.update()
    # Apply transform
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    root.rotation_euler = (0, 0, math.radians(rot_deg))
    root.location = Vector(position)
    bpy.context.view_layer.update()
    return root


def setup_gpu(scene, gpu_device):
    scene.render.engine = "CYCLES"
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    cprefs.compute_device_type = "OPTIX"
    cprefs.get_devices()
    backend_devices = [d for d in cprefs.devices if d.type == "OPTIX"]
    for d in cprefs.devices:
        d.use = False
    if 0 <= gpu_device < len(backend_devices):
        backend_devices[gpu_device].use = True
    elif backend_devices:
        backend_devices[0].use = True
    scene.cycles.device = "GPU"


def main():
    args = parse_args(sys.argv)
    # Open scene
    bpy.ops.wm.open_mainfile(filepath=args.scene_path)
    scene = bpy.context.scene

    # Place object
    import_and_place_object(
        scene, args.object_path,
        args.obj_pos, args.obj_rot, args.obj_scale,
    )

    # Setup camera
    cam = scene.camera
    if cam is None:
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                cam = obj
                break
    if cam is None:
        cam_data = bpy.data.cameras.new("IterCam")
        cam = bpy.data.objects.new("IterCam", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam
    # Clear constraints/animation
    if cam.parent:
        cam.parent = None
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    if cam.animation_data:
        cam.animation_data_clear()

    cam.data.lens = args.focal_length
    cam.data.sensor_width = args.sensor_width

    cam_pos = Vector(args.cam_pos)
    cam_target = Vector(args.cam_target)
    cam.location = cam_pos
    rot = look_at_matrix(cam_pos, cam_target)
    cam.rotation_euler = rot.to_euler()

    # Render settings
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = args.samples
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.05
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPTIX"
    scene.render.use_persistent_data = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4

    # GPU
    setup_gpu(scene, args.gpu_device)

    # Render
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {out_path}")


if __name__ == "__main__":
    main()
