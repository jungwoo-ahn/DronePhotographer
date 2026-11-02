"""Render eval subjects for compositing over splat backgrounds (runs inside Blender).

  blender -b -P blender_subjects.py -- JOBS.json [--cpu] [--samples N] [--only i j ...]

Per job: a fresh scene with the subject imported exactly as the training renders import it
(shared `render_object.import_object` + bottom-centre to origin), turned by `rotation_z_deg`,
lit only by the splat's own 360 deg environment map, standing on a shadow-catcher plane at z=0,
seen through the job's camera (24 mm on 12.8 x 9.6 mm, as v7). Writes RGBA (straight alpha;
the shadow catcher's shadow arrives as black with partial alpha) and a Z pass as OpenEXR.
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
jobs = json.loads(Path(argv[0]).read_text())
use_cpu = "--cpu" in argv
samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 64
only = None
if "--only" in argv:
    only = {int(x) for x in argv[argv.index("--only") + 1:] if x.lstrip("-").isdigit()}

SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, SHARED)
import render_object as ro  # noqa: E402  (shared importer: same placement code as the training data)


def setup_device(scene):
    scene.render.engine = "CYCLES"
    if use_cpu:
        scene.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue
        devs = [d for d in prefs.devices if d.type == backend]
        if devs:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return backend
    scene.cycles.device = "CPU"
    return "CPU"


for job in jobs:
    if only is not None and job["index"] not in only:
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    dev = setup_device(scene)
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    W, H = job["camera"]["res"]
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100

    objs = ro.import_object(ro.pick_first_file(Path(job["blend_dir"]), ro.PICK_EXTS_OBJ))
    root = ro.parent_imported_objects(objs)
    ro.move_group_bottom_center_to_origin(root, objs)
    root.rotation_euler = (0.0, 0.0, math.radians(job["rotation_z_deg"]))
    root.location = (0.0, 0.0, 0.0)

    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    bpy.context.active_object.is_shadow_catcher = True

    world = bpy.data.worlds.new("env")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(job["env"])
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = float(job.get("env_strength", 1.0))

    cd = bpy.data.cameras.new("cam")
    c = job["camera"]
    cd.lens, cd.sensor_width, cd.sensor_height = c["lens_mm"], c["sensor_w_mm"], c["sensor_h_mm"]
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start = 0.05
    cam = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.matrix_world = Matrix.Translation(Vector(c["location"])) @ Matrix(c["rotation"]).to_4x4()

    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = str(Path(job["fg_depth"]).parent)
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_slots[0].path = Path(job["fg_depth"]).stem + "_"
    tree.links.new(rl.outputs["Depth"], fo.inputs[0])

    s = scene.render.image_settings
    s.file_format, s.color_mode, s.color_depth = "PNG", "RGBA", "16"
    scene.render.filepath = job["fg"]
    bpy.ops.render.render(write_still=True)
    print(f"[subjects] {job['scene']} {job['tag']} rendered on {dev}", flush=True)
