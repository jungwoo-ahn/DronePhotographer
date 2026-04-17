"""Render preview thumbnails for all scenes and objects.

Produces:
  debug/scenes/<scene_name>.png   — scene rendered from its own camera (or auto)
  debug/objects/<object_name>.png — object on neutral gray background

Usage:
    python scripts/render_asset_previews.py --config configs/vlm_placement.json
    python scripts/render_asset_previews.py --config configs/vlm_placement.json --only scenes
    python scripts/render_asset_previews.py --config configs/vlm_placement.json --only objects
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assets import (
    OBJECT_EXTS,
    SCENE_EXTS,
    discover_files,
    find_scene_blend,
    object_name,
    scene_name,
)
from src.blender.gpu import gpu_config_snippet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def render_scene_preview(blender_bin: str, blend_path: Path, output_path: str,
                         resolution: int = 512, samples: int = 32, timeout: int = 120,
                         gpu_backend: str = "AUTO", gpu_devices: list[int] | None = None):
    """Render a scene using its built-in camera (or a default view)."""
    gpu_snippet = gpu_config_snippet(gpu_backend, gpu_devices)
    script = f"""
import bpy, math
from mathutils import Vector

scene = bpy.context.scene

# Use existing camera if available, otherwise create one
cam = scene.camera
if cam is None:
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            scene.camera = obj
            cam = obj
            break

if cam is None:
    # Create camera looking at scene center
    cam_data = bpy.data.cameras.new('PreviewCam')
    cam = bpy.data.objects.new('PreviewCam', cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if meshes:
        min_v = Vector((1e9, 1e9, 1e9))
        max_v = Vector((-1e9, -1e9, -1e9))
        for o in meshes:
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                min_v = Vector((min(min_v.x, w.x), min(min_v.y, w.y), min(min_v.z, w.z)))
                max_v = Vector((max(max_v.x, w.x), max(max_v.y, w.y), max(max_v.z, w.z)))
        center = (min_v + max_v) / 2
        diag = (max_v - min_v).length
        cam.location = center + Vector((diag * 0.5, -diag * 0.5, diag * 0.4))
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Fallback lighting: if no lights and no world shader, add a neutral environment
_has_lights = any(o.type == 'LIGHT' for o in bpy.data.objects)
_world = scene.world
_has_world = False
if _world and _world.use_nodes:
    for _n in _world.node_tree.nodes:
        if _n.type in ('TEX_SKY', 'TEX_ENVIRONMENT', 'TEX_IMAGE', 'BACKGROUND'):
            if _n.type == 'BACKGROUND':
                _c = _n.inputs['Color']
                if _c.is_linked or sum(_c.default_value[:3]) > 0.01:
                    _has_world = True
                    break
            else:
                _has_world = True
                break

if not _has_lights and not _has_world:
    print('FALLBACK_LIGHTING: no lights or world shader found, adding environment light')
    _fb_world = bpy.data.worlds.new('FallbackWorld')
    scene.world = _fb_world
    _fb_world.use_nodes = True
    _fb_tree = _fb_world.node_tree
    _fb_tree.nodes.clear()
    _fb_bg = _fb_tree.nodes.new('ShaderNodeBackground')
    _fb_bg.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    _fb_bg.inputs['Strength'].default_value = 1.0
    _fb_out = _fb_tree.nodes.new('ShaderNodeOutputWorld')
    _fb_tree.links.new(_fb_bg.outputs['Background'], _fb_out.inputs['Surface'])

scene.render.resolution_x = {resolution}
scene.render.resolution_y = {resolution}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = '{output_path}'
scene.render.engine = 'CYCLES'
scene.cycles.samples = {samples}
scene.cycles.use_denoising = True
{gpu_snippet}
bpy.ops.render.render(write_still=True)
print('RENDER_OK')
"""
    cmd = [blender_bin, "--background", str(blend_path), "--python-expr", script]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return "RENDER_OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def render_object_preview(blender_bin: str, object_path: Path, output_path: str,
                          resolution: int = 512, samples: int = 32, timeout: int = 120,
                          gpu_backend: str = "AUTO", gpu_devices: list[int] | None = None):
    """Render an object on a neutral background."""
    obj_str = str(object_path).replace("'", "\\'")
    ext = object_path.suffix.lower()

    if ext in (".glb", ".gltf"):
        import_cmd = f"bpy.ops.import_scene.gltf(filepath='{obj_str}')"
    elif ext == ".fbx":
        import_cmd = f"bpy.ops.import_scene.fbx(filepath='{obj_str}')"
    elif ext == ".obj":
        import_cmd = f"""
try:
    bpy.ops.wm.obj_import(filepath='{obj_str}')
except:
    bpy.ops.import_scene.obj(filepath='{obj_str}')
"""
    elif ext == ".blend":
        import_cmd = f"""
with bpy.data.libraries.load('{obj_str}', link=False) as (data_from, data_to):
    data_to.objects = list(data_from.objects)
for obj in data_to.objects:
    if obj is not None:
        bpy.context.collection.objects.link(obj)
"""
    else:
        return False

    gpu_snippet = gpu_config_snippet(gpu_backend, gpu_devices)
    script = f"""
import bpy, math
from mathutils import Vector

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import object
before = set(o.name for o in bpy.data.objects)
{import_cmd}
imported = [o for o in bpy.data.objects if o.name not in before]

if not imported:
    print('NO_OBJECTS')
else:
    # Parent under root for easy scaling
    root = bpy.data.objects.new('Root', None)
    bpy.context.collection.objects.link(root)
    imp_set = set(imported)
    for o in imported:
        if o.parent not in imp_set:
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()
    bpy.context.view_layer.update()

    # Compute bbox
    min_v = Vector((1e9, 1e9, 1e9))
    max_v = Vector((-1e9, -1e9, -1e9))
    for o in imported:
        if not hasattr(o, 'bound_box'):
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            min_v = Vector((min(min_v.x, w.x), min(min_v.y, w.y), min(min_v.z, w.z)))
            max_v = Vector((max(max_v.x, w.x), max(max_v.y, w.y), max(max_v.z, w.z)))

    size = max_v - min_v
    diag = size.length

    # Normalize: scale so largest dimension = 2.0 units
    max_dim = max(size.x, size.y, size.z)
    if max_dim > 0.001:
        s = 2.0 / max_dim
        root.scale = (s, s, s)
        bpy.context.view_layer.update()
        # Recompute bbox after scale
        min_v = Vector((1e9, 1e9, 1e9))
        max_v = Vector((-1e9, -1e9, -1e9))
        for o in imported:
            if not hasattr(o, 'bound_box'):
                continue
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                min_v = Vector((min(min_v.x, w.x), min(min_v.y, w.y), min(min_v.z, w.z)))
                max_v = Vector((max(max_v.x, w.x), max(max_v.y, w.y), max(max_v.z, w.z)))
        size = max_v - min_v
        diag = size.length

    # Move bottom to z=0
    root.location.z -= min_v.z
    bpy.context.view_layer.update()
    min_v.z = 0
    max_v.z = size.z

    center = (min_v + max_v) / 2

    # Camera
    cam_data = bpy.data.cameras.new('Cam')
    cam_data.lens = 50  # tighter framing
    cam = bpy.data.objects.new('Cam', cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    dist = max(diag * 2.0, 0.5)  # ensure minimum distance
    look_at = center + Vector((0, 0, size.z * 0.15))  # aim slightly above center
    cam.location = look_at + Vector((dist * 0.4, -dist * 0.6, dist * 0.35))
    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Lighting
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    bg = tree.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    output = tree.nodes.new('ShaderNodeOutputWorld')
    tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=diag * 5, location=(center.x, center.y, min_v.z))

    scene = bpy.context.scene
    scene.render.resolution_x = {resolution}
    scene.render.resolution_y = {resolution}
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = '{output_path}'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = {samples}
    scene.cycles.use_denoising = True
{gpu_snippet}
    bpy.ops.render.render(write_still=True)
    print('RENDER_OK')
"""
    cmd = [blender_bin, "--background", "--python-expr", script]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return "RENDER_OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", choices=["scenes", "objects"], default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--gpu_backend", default="AUTO",
                   choices=["AUTO", "OPTIX", "CUDA", "CPU"],
                   help="GPU backend for Cycles (default: AUTO = try OPTIX then CUDA)")
    p.add_argument("--gpu_devices", type=int, nargs="*", default=None,
                   help="GPU device indices to use (default: all available)")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    blender_bin = config["blender"]["binary"]
    scenes_dir = Path(config["assets"]["scenes_dir"])
    objects_dir = Path(config["assets"]["objects_dir"])

    debug_dir = Path("debug")
    scenes_out = debug_dir / "scenes"
    objects_out = debug_dir / "objects"
    scenes_out.mkdir(parents=True, exist_ok=True)
    objects_out.mkdir(parents=True, exist_ok=True)

    # Scenes
    if args.only != "objects":
        scene_blends = []
        # Standalone .blend files
        for f in sorted(scenes_dir.glob("*.blend")):
            scene_blends.append(f)
        # .blend inside subdirectories
        try:
            for d in sorted(scenes_dir.iterdir()):
                if d.is_dir():
                    b = find_scene_blend(d)
                    if b:
                        scene_blends.append(b)
        except PermissionError:
            pass
        scene_blends = sorted(set(scene_blends))

        log.info(f"Rendering {len(scene_blends)} scene previews...")
        ok = 0
        for i, blend in enumerate(scene_blends, 1):
            name = scene_name(blend)
            out = str(scenes_out / f"{name}.png")
            if Path(out).exists():
                log.info(f"  [{i}/{len(scene_blends)}] SKIP {name}")
                ok += 1
                continue
            log.info(f"  [{i}/{len(scene_blends)}] {name}...")
            if render_scene_preview(blender_bin, blend, out,
                                    resolution=args.resolution,
                                    samples=args.samples,
                                    timeout=args.timeout,
                                    gpu_backend=args.gpu_backend,
                                    gpu_devices=args.gpu_devices):
                ok += 1
                log.info(f"    OK")
            else:
                log.warning(f"    FAILED")
        log.info(f"Scenes: {ok}/{len(scene_blends)} rendered")

    # Objects
    if args.only != "scenes":
        obj_files = []
        for item in sorted(objects_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in OBJECT_EXTS:
                obj_files.append(item)
            elif item.is_dir():
                for ext in OBJECT_EXTS:
                    for f in sorted(item.rglob(f"*{ext}")):
                        obj_files.append(f)
                        break  # one per directory
        obj_files = sorted(set(obj_files))

        log.info(f"Rendering {len(obj_files)} object previews...")
        ok = 0
        for i, obj_path in enumerate(obj_files, 1):
            name = object_name(obj_path)
            out = str(objects_out / f"{name}.png")
            if Path(out).exists():
                log.info(f"  [{i}/{len(obj_files)}] SKIP {name}")
                ok += 1
                continue
            log.info(f"  [{i}/{len(obj_files)}] {name}...")
            if render_object_preview(blender_bin, obj_path, out,
                                     resolution=args.resolution,
                                     samples=args.samples,
                                     timeout=args.timeout,
                                     gpu_backend=args.gpu_backend,
                                     gpu_devices=args.gpu_devices):
                ok += 1
                log.info(f"    OK")
            else:
                log.warning(f"    FAILED")
        log.info(f"Objects: {ok}/{len(obj_files)} rendered")


if __name__ == "__main__":
    main()
