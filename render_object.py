
import argparse
import json
import math
import random
import sys
from datetime import datetime
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

sys.path.append('.')
from src.scenes.scene import open_scene, set_nishita_sky

#Priority: --object_position > --auto_place_object > --object_name


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render random views of an object in a Blender scene.")
    parser.add_argument("--input_scene", default="assets/Koky_LuxuryHouse_0.blend")     #ahn, 260309: all assets moved to /home/nas5/jungwooahn/datasets/DronePhotos
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name")
    parser.add_argument("--output_run_dir", help="explicit output run directory (overrides auto-generated path)")
    parser.add_argument("--sky_strength", type=float, default=0.1)
    parser.add_argument("--object_name", help="object name in scene (e.g., GEO-snowman-head.001)")
    parser.add_argument("--object_position", nargs=3, type=float)
    parser.add_argument("--auto_place_object", action="store_true")
    parser.add_argument("--input_object", help="path to object file or folder (used with --auto_place_object or --object_position)")
    parser.add_argument("--rotation_z_deg", type=float, default=0.0, help="Z-axis rotation in degrees for imported object")
    parser.add_argument(
        "--object_rotation_xyz_rad",
        nargs=3,
        type=float,
        default=None,
        help="full Euler XYZ rotation in radians for imported object (overrides --rotation_z_deg when set)",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="uniform scale for imported object")
    parser.add_argument(
        "--scene_scale",
        type=float,
        default=1.0,
        help="uniform scale applied to the scene root before placing the object (matches placement-pipeline scene_scale)",
    )
    parser.add_argument("--use_aabb_center", action="store_true", help="use AABB center as camera orbit center instead of object position")
    parser.add_argument("--hemisphere", action="store_true")
    parser.add_argument("--camera_radius_range", nargs=2, type=float, default=[2.0, 10.0])
    parser.add_argument("--camera_direction_offsets", nargs=3, type=float, default=[30.0, 30.0, 30.0], help="yaw pitch roll offsets in degrees")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--disable_gpu", action="store_true", help="force CPU rendering; by default tries OPTIX/CUDA")
    parser.add_argument(
        "--gpu_backend",
        choices=["AUTO", "OPTIX", "CUDA"],
        default="AUTO",
        help="preferred Cycles GPU backend",
    )
    parser.add_argument(
        "--gpu_devices",
        nargs="+",
        type=int,
        help="optional GPU device indices for selected backend (default: all available)",
    )
    parser.add_argument("--seed", type=int, default=721, help="random seed for camera sampling")
    parser.add_argument("--keep_camera_constraints", action="store_true", help="keep existing camera constraints/animation")
    # Camera: DJI Mini 5 Pro specs
    parser.add_argument("--focal_length", type=float, default=24.0)
    parser.add_argument("--sensor_width", type=float, default=12.8)
    parser.add_argument("--sensor_height", type=float, default=9.6)
    parser.add_argument("--resolution", nargs=2, type=int, default=[1024, 768])
    parser.add_argument("--samples", type=int, default=64, help="Cycles sample count per image")
    parser.add_argument("--adaptive_sampling", action="store_true", help="enable Cycles adaptive sampling")
    parser.add_argument("--adaptive_threshold", type=float, default=0.01, help="adaptive sampling noise threshold")
    parser.add_argument("--no_denoise", action="store_true", help="disable OptiX denoiser (enabled by default)")
    parser.add_argument("--max_bounces", type=int, default=4)
    parser.add_argument("--diffuse_bounces", type=int, default=2)
    parser.add_argument("--glossy_bounces", type=int, default=2)
    parser.add_argument("--transmission_bounces", type=int, default=2)
    parser.add_argument("--volume_bounces", type=int, default=0)
    parser.add_argument("--transparent_max_bounces", type=int, default=4)
    parser.add_argument(
        "--persistent_data",
        action="store_true",
        help="reuse render data across frames to reduce per-frame overhead",
    )
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=4,
        help="number of CPU threads for Blender (1-4 recommended for GPU rendering)",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="total number of parallel workers (each gets a separate GPU)",
    )
    parser.add_argument(
        "--worker_index",
        type=int,
        default=0,
        help="this worker's index (0-based), determines which images to render",
    )
    parser.add_argument("--render_depth", action="store_true", help="render depth map alongside RGB")
    parser.add_argument("--depth_format", choices=["OPEN_EXR", "PNG"], default="PNG", help="depth output format (OPEN_EXR recommended)")
    parser.add_argument("--depth_max", type=float, default=15.0, help="max depth for PNG normalization (ignored for EXR)")
    split = argv.index("--") + 1 if "--" in argv else len(argv)
    return parser.parse_args(argv[split:])


def ensure_camera(scene):
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam = bpy.data.objects.new("RenderCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_sky(strength):
    set_nishita_sky(float(strength))


def set_sky_if_no_native_lighting(strength, imported_names=None):
    """Scene-adaptive sky lighting (mirrors scripts/vlm_place_worker.py:455-473).

    Only adds Nishita sky when the BLEND scene has no LIGHT objects AND no
    sky / environment / non-black background in its world shader. Prevents
    overwriting a scene's native HDRI / sun + bake interior scenes that
    already ship their own lighting.
    """
    imported_names = imported_names or set()
    has_lights = any(
        o.type == "LIGHT" and o.name not in imported_names
        for o in bpy.data.objects
    )
    has_world = False
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type in ("TEX_SKY", "TEX_ENVIRONMENT", "TEX_IMAGE"):
                has_world = True
                break
            if node.type == "BACKGROUND":
                c = node.inputs["Color"]
                if c.is_linked or sum(c.default_value[:3]) > 0.01:
                    has_world = True
                    break
    if not has_lights and not has_world:
        print("No native lighting found, adding Nishita sky")
        set_nishita_sky(float(strength))
        return True
    print(f"Using scene lighting (lights={has_lights}, world={has_world})")
    return False


def apply_scene_scale(scene, scale_factor):
    """Uniform-scale the scene by ``scale_factor`` and compensate light intensity.

    Scales every root-level object's location and scale, then multiplies
    non-sun light energy by scale_factor**2 (point/area/spot lights follow an
    inverse-square law, so shrinking distances by s requires energy * s^2 to
    keep illuminance constant). Sun lights use directional irradiance and are
    left untouched. Mirrors scripts/vlm_place_worker.py:apply_scene_scale.
    """
    s = float(scale_factor)
    if abs(s - 1.0) < 1e-9:
        return
    for obj in scene.objects:
        if obj.parent is None:
            obj.location = (obj.location.x * s, obj.location.y * s, obj.location.z * s)
            obj.scale = (obj.scale.x * s, obj.scale.y * s, obj.scale.z * s)
    energy_factor = s * s
    for light in bpy.data.lights:
        if light.type == "SUN":
            continue
        light.energy = light.energy * energy_factor
    bpy.context.view_layer.update()


def set_render_settings(scene, resolution, args):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_persistent_data = bool(args.persistent_data)
    scene.render.film_transparent = False

    cycles = scene.cycles
    cycles.samples = int(args.samples)
    cycles.max_bounces = int(args.max_bounces)
    cycles.diffuse_bounces = int(args.diffuse_bounces)
    cycles.glossy_bounces = int(args.glossy_bounces)
    cycles.transmission_bounces = int(args.transmission_bounces)
    cycles.volume_bounces = int(args.volume_bounces)
    cycles.transparent_max_bounces = int(args.transparent_max_bounces)
    if hasattr(cycles, "use_adaptive_sampling"):
        cycles.use_adaptive_sampling = bool(args.adaptive_sampling)
    if hasattr(cycles, "adaptive_threshold"):
        cycles.adaptive_threshold = float(args.adaptive_threshold)

    # Denoiser
    if hasattr(cycles, "use_denoising"):
        cycles.use_denoising = not args.no_denoise
        if not args.no_denoise:
            cycles.denoiser = "OPTIX"

    if hasattr(scene.render, "threads_mode"):
        scene.render.threads_mode = "FIXED"
        scene.render.threads = max(1, int(args.blender_threads))


def configure_depth_output(
    scene: bpy.types.Scene,
    depth_dir: Path,
    depth_format: str,
    depth_max: float,
) -> bpy.types.Node:
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    output = tree.nodes.new("CompositorNodeOutputFile")
    output.base_path = str(depth_dir)
    output.format.color_mode = "BW"

    if depth_format == "OPEN_EXR":
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "32"
        tree.links.new(render_layers.outputs["Depth"], output.inputs[0])
        return output

    map_node = tree.nodes.new("CompositorNodeMapRange")
    map_node.inputs["From Min"].default_value = 0.0
    map_node.inputs["From Max"].default_value = depth_max
    map_node.inputs["To Min"].default_value = 1.0
    map_node.inputs["To Max"].default_value = 0.0
    if hasattr(map_node, "use_clamp"):
        map_node.use_clamp = True
    output.format.file_format = "PNG"
    output.format.color_depth = "16"
    tree.links.new(render_layers.outputs["Depth"], map_node.inputs["Value"])
    tree.links.new(map_node.outputs["Value"], output.inputs[0])
    return output


def prepare_camera(cam: bpy.types.Object, keep_constraints: bool) -> None:
    if keep_constraints:
        return
    if cam.parent is not None:
        cam.parent = None
    if cam.constraints:
        for constraint in list(cam.constraints):
            cam.constraints.remove(constraint)
    if cam.animation_data is not None:
        cam.animation_data_clear()


def object_position_from_name(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object '{object_name}' not found in the scene.")
    return obj.location.copy()


REPO_ROOT = Path(__file__).resolve().parent
PICK_EXTS_OBJ = [".glb", ".gltf", ".fbx", ".obj", ".blend"]
MAX_SLOPE_DEG = 12.0
HEIGHT_TOL = 0.08
FOOTPRINT_MARGIN = 0.15
GRID_STEP_BASE = 1.0
CELL_SIZE = 1.0
MAX_GRID_SAMPLES = 20000
SAMPLES_PER_SIDE = 3


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def pick_first_file(path: Path, exts: list[str]) -> Path | None:
    path = resolve_path(path)
    if path.is_file():
        return path
    if not path.exists():
        return None
    for ext in exts:
        for cand in sorted(path.rglob(f"*{ext}")):
            return cand
    return None


def import_object(obj_file: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    ext = obj_file.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(obj_file))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(obj_file))
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(obj_file))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(obj_file), link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported object file: {obj_file}")
    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    return imported


def parent_imported_objects(imported: list[bpy.types.Object]) -> bpy.types.Object:
    root = bpy.data.objects.new("AutoPlacedRoot", None)
    bpy.context.scene.collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent in imported_set:
            continue
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    return root


def get_world_bbox(objs: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in objs:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, world.x)
            min_v.y = min(min_v.y, world.y)
            min_v.z = min(min_v.z, world.z)
            max_v.x = max(max_v.x, world.x)
            max_v.y = max(max_v.y, world.y)
            max_v.z = max(max_v.z, world.z)
            found = True
    if not found:
        return None
    return min_v, max_v


def move_group_bottom_center_to_origin(root: bpy.types.Object, imported: list[bpy.types.Object]) -> None:
    bbox = get_world_bbox(imported)
    if bbox is None:
        return
    min_v, max_v = bbox
    bottom_center = Vector(((min_v.x + max_v.x) * 0.5, (min_v.y + max_v.y) * 0.5, min_v.z))
    root.location = root.location - bottom_center


def set_hidden(objs: list[bpy.types.Object], hidden: bool) -> list[tuple[bpy.types.Object, bool, bool]]:
    prev = []
    for obj in objs:
        prev.append((obj, bool(obj.hide_viewport), bool(obj.hide_render)))
        try:
            obj.hide_set(hidden)
        except Exception:
            obj.hide_viewport = hidden
        obj.hide_render = hidden
    return prev


def restore_hidden(prev: list[tuple[bpy.types.Object, bool, bool]]) -> None:
    for obj, hv, hr in prev:
        try:
            obj.hide_set(hv)
        except Exception:
            obj.hide_viewport = hv
        obj.hide_render = hr


def iter_grid(x_min: float, x_max: float, y_min: float, y_max: float, step: float):
    if step <= 0:
        return
    nx = max(1, int(math.floor((x_max - x_min) / step)) + 1)
    ny = max(1, int(math.floor((y_max - y_min) / step)) + 1)
    for i in range(nx):
        x = x_min + i * step
        for j in range(ny):
            y = y_min + j * step
            yield x, y


def compute_grid_step(bounds_min: Vector, bounds_max: Vector, desired_step: float) -> float:
    width = max(0.01, bounds_max.x - bounds_min.x)
    depth = max(0.01, bounds_max.y - bounds_min.y)
    area = width * depth
    min_step = math.sqrt(area / MAX_GRID_SAMPLES)
    return max(desired_step, min_step)


def area_is_flat(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    center: Vector,
    obj_size: Vector,
    z_top: float,
    slope_cos: float,
) -> bool:
    half_x = obj_size.x * 0.5 + FOOTPRINT_MARGIN
    half_y = obj_size.y * 0.5 + FOOTPRINT_MARGIN
    heights = []
    if SAMPLES_PER_SIDE <= 1:
        samples = [center]
    else:
        samples = []
        for i in range(SAMPLES_PER_SIDE):
            for j in range(SAMPLES_PER_SIDE):
                tx = -half_x + (2 * half_x) * (i / (SAMPLES_PER_SIDE - 1))
                ty = -half_y + (2 * half_y) * (j / (SAMPLES_PER_SIDE - 1))
                samples.append(Vector((center.x + tx, center.y + ty, center.z)))
    for sample in samples:
        origin = Vector((sample.x, sample.y, z_top))
        direction = Vector((0.0, 0.0, -1.0))
        hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or obj is None:
            return False
        if normal.z < slope_cos:
            return False
        heights.append(loc.z)
    if not heights:
        return False
    if max(heights) - min(heights) > HEIGHT_TOL:
        return False
    return True


def find_best_flat_location(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    bounds_min: Vector,
    bounds_max: Vector,
    obj_size: Vector,
    ignore_names: set[str],
) -> tuple[Vector, Vector] | None:
    slope_cos = math.cos(math.radians(MAX_SLOPE_DEG))
    obj_radius = max(obj_size.x, obj_size.y) * 0.5

    x_min = bounds_min.x + obj_radius
    x_max = bounds_max.x - obj_radius
    y_min = bounds_min.y + obj_radius
    y_max = bounds_max.y - obj_radius
    if x_min >= x_max or y_min >= y_max:
        x_min, x_max = bounds_min.x, bounds_max.x
        y_min, y_max = bounds_min.y, bounds_max.y

    step = compute_grid_step(bounds_min, bounds_max, max(GRID_STEP_BASE, obj_radius * 0.5))
    z_top = bounds_max.z + max(5.0, obj_size.z * 2.0)

    cells: dict[tuple[int, int], dict[str, object]] = {}
    for x, y in iter_grid(x_min, x_max, y_min, y_max, step):
        origin = Vector((x, y, z_top))
        direction = Vector((0.0, 0.0, -1.0))
        hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
        if not hit or obj is None:
            continue
        if obj.name in ignore_names:
            continue
        if normal.z < slope_cos:
            continue
        ix = int(math.floor((loc.x - bounds_min.x) / CELL_SIZE))
        iy = int(math.floor((loc.y - bounds_min.y) / CELL_SIZE))
        key = (ix, iy)
        cell = cells.get(key)
        if cell is None:
            cell = {"count": 0, "sum_loc": Vector((0.0, 0.0, 0.0)), "sum_n": Vector((0.0, 0.0, 0.0))}
            cells[key] = cell
        cell["count"] += 1
        cell["sum_loc"] += loc
        cell["sum_n"] += normal

    if not cells:
        return None

    ranked = sorted(cells.values(), key=lambda c: c["count"], reverse=True)
    for cell in ranked:
        count = cell["count"]
        if count <= 0:
            continue
        center = cell["sum_loc"] / count
        normal = cell["sum_n"].normalized()
        if area_is_flat(scene, depsgraph, center, obj_size, z_top, slope_cos):
            return center, normal

    return None


def auto_place_object(scene: bpy.types.Scene, input_object: str) -> Vector:
    obj_file = pick_first_file(Path(input_object), PICK_EXTS_OBJ)
    if obj_file is None:
        raise FileNotFoundError(f"Object not found: {input_object}")
    imported_objs = import_object(obj_file)
    if not imported_objs:
        raise RuntimeError("No objects were imported.")

    root = parent_imported_objects(imported_objs)
    move_group_bottom_center_to_origin(root, imported_objs)
    bpy.context.view_layer.update()

    bbox = get_world_bbox(imported_objs)
    if bbox is None:
        raise RuntimeError("Could not compute bounding box for imported objects.")
    obj_min, obj_max = bbox
    obj_size = obj_max - obj_min

    hidden_state = set_hidden(imported_objs, True)
    try:
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        scene_meshes = [
            obj
            for obj in scene.objects
            if obj.type == "MESH" and obj.name not in {o.name for o in imported_objs}
        ]
        bounds = get_world_bbox(scene_meshes)
        if bounds is None:
            raise RuntimeError("No mesh geometry found in the scene.")
        bounds_min, bounds_max = bounds
        found = find_best_flat_location(
            scene,
            depsgraph,
            bounds_min,
            bounds_max,
            obj_size,
            ignore_names={o.name for o in imported_objs},
        )
    finally:
        restore_hidden(hidden_state)
        bpy.context.view_layer.update()

    if found is None:
        bounds_min, bounds_max = get_world_bbox([obj for obj in scene.objects if obj.type == "MESH"]) or (
            Vector((0.0, 0.0, 0.0)),
            Vector((0.0, 0.0, 0.0)),
        )
        target_loc = Vector(((bounds_min.x + bounds_max.x) * 0.5, (bounds_min.y + bounds_max.y) * 0.5, bounds_max.z))
        target_normal = Vector((0.0, 0.0, 1.0))
    else:
        target_loc, target_normal = found

    up = Vector((0.0, 0.0, 1.0))
    rot = up.rotation_difference(target_normal)
    root.rotation_euler = rot.to_euler()
    root.location = target_loc + target_normal * 0.02
    bpy.context.view_layer.update()
    return root.location.copy()


def place_imported_object(scene, input_object, position, rotation_z_deg=0.0, scale=1.0,
                          rotation_xyz_rad=None):
    """Import an external object and place it at a specified position with rotation and scale.

    rotation_xyz_rad: optional (rx, ry, rz) Euler XYZ in radians; takes precedence
    over rotation_z_deg when provided.

    Returns (obj_position, root_object_name).
    """
    obj_file = pick_first_file(Path(input_object), PICK_EXTS_OBJ)
    if obj_file is None:
        raise FileNotFoundError(f"Object not found: {input_object}")
    imported_objs = import_object(obj_file)
    if not imported_objs:
        raise RuntimeError("No objects were imported.")

    root = parent_imported_objects(imported_objs)
    move_group_bottom_center_to_origin(root, imported_objs)
    bpy.context.view_layer.update()

    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    if rotation_xyz_rad is not None:
        rx, ry, rz = (float(v) for v in rotation_xyz_rad)
        root.rotation_euler = (rx, ry, rz)
    else:
        root.rotation_euler = (0, 0, radians(rotation_z_deg))
    root.location = Vector(position)
    bpy.context.view_layer.update()

    return root.location.copy(), root.name


def random_direction(hemisphere):
    u = random.random()
    v = random.random()
    theta = 2 * pi * u
    z = v if hemisphere else 2 * v - 1
    r = sqrt(max(0.0, 1.0 - z * z))
    return Vector((r * cos(theta), r * sin(theta), z))


def look_at_matrix(cam_pos, target, world_up=Vector((0.0, 0.0, 1.0))):
    # Blender camera convention:
    # local +X: right, local +Y: up, local -Z: forward (view direction)
    forward = (target - cam_pos).normalized()
    # Choose right as forward x up so image stays upright to world_up.
    right = forward.cross(world_up)
    if right.length < 1e-6:
        right = forward.cross(Vector((0.0, 1.0, 0.0)))
    right.normalize()
    up = right.cross(forward).normalized()
    return Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )


def sample_pose(obj_pos, radius_range, offsets_deg, hemisphere):
    r_min, r_max = sorted(radius_range)
    direction = random_direction(hemisphere)
    radius = random.uniform(r_min, r_max)
    cam_pos = obj_pos + direction * radius
    base_rot = look_at_matrix(cam_pos, obj_pos)
    base_forward = (obj_pos - cam_pos).normalized()
    yaw_range, pitch_range, roll_range = offsets_deg
    yaw = random.uniform(-yaw_range, yaw_range)
    pitch = random.uniform(-pitch_range, pitch_range)
    roll = random.uniform(-roll_range, roll_range)
    offset = Euler((radians(roll), radians(pitch), radians(yaw)), "XYZ").to_matrix()
    rot_matrix = base_rot @ offset
    final_forward = (rot_matrix @ Vector((0.0, 0.0, -1.0))).normalized()
    final_up = (rot_matrix @ Vector((0.0, 1.0, 0.0))).normalized()
    return {
        "cam_pos": cam_pos,
        "radius": radius,
        "base_forward": base_forward,
        "base_up": (base_rot @ Vector((0.0, 1.0, 0.0))).normalized(),
        "rot_matrix": rot_matrix,
        "offsets": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "final_forward": final_forward,
        "final_up": final_up,
    }


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def configure_gpu(scene, disable_gpu=False, gpu_backend="AUTO", gpu_devices=None):
    scene.render.engine = "CYCLES"
    if disable_gpu:
        scene.cycles.device = "CPU"
        return {"device": "CPU", "devices": []}
    cprefs_addon = bpy.context.preferences.addons.get("cycles")
    if cprefs_addon is None:
        scene.cycles.device = "CPU"
        return {"device": "CPU", "devices": []}
    cprefs = cprefs_addon.preferences
    chosen = {"device": "CPU", "devices": []}
    backend_order = ("OPTIX", "CUDA") if gpu_backend == "AUTO" else (gpu_backend,)
    for dtype in backend_order:
        try:
            cprefs.compute_device_type = dtype
            cprefs.get_devices()
        except Exception:
            continue
        backend_devices = [d for d in cprefs.devices if d.type == dtype]
        if not backend_devices:
            continue
        selected_indices = set()
        if gpu_devices:
            valid = [idx for idx in gpu_devices if 0 <= idx < len(backend_devices)]
            selected_indices = set(valid)
        for d in cprefs.devices:
            d.use = False
        for idx, d in enumerate(backend_devices):
            d.use = idx in selected_indices if selected_indices else True
        scene.cycles.device = "GPU"
        chosen = {"device": dtype, "devices": [d.name for d in backend_devices if d.use]}
        break
    else:
        try:
            cprefs.compute_device_type = "NONE"
            cprefs.get_devices()
        except Exception:
            pass
        scene.cycles.device = "CPU"
    return chosen


def main(argv):
    args = parse_args(argv)
    random.seed(args.seed)
    blend_path = Path(args.input_scene)
    scene_stem = blend_path.stem
    if args.output_run_dir:
        run_dir = Path(args.output_run_dir)
        timestamp = run_dir.name.split("_")[-1] if "_" in run_dir.name else datetime.now().strftime("%y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        folder_name = f"{scene_stem}_{timestamp}"
        if args.run_name:
            folder_name = f"{scene_stem}_{args.run_name}_{timestamp}"
        run_dir = Path(args.output_dir) / folder_name
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = None
    depth_output_node = None

    scene = open_scene(str(blend_path))
    camera = ensure_camera(scene)
    prepare_camera(camera, args.keep_camera_constraints)
    if args.focal_length is not None:
        camera.data.lens = args.focal_length
    if args.sensor_width is not None:
        camera.data.sensor_width = args.sensor_width
    if args.sensor_height is not None:
        camera.data.sensor_height = args.sensor_height
    if args.sky_strength > 0:
        set_sky(args.sky_strength)
    device_info = configure_gpu(scene, args.disable_gpu, args.gpu_backend, args.gpu_devices)
    set_render_settings(scene, args.resolution, args)
    if args.render_depth:
        depth_dir = run_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_output_node = configure_depth_output(scene, depth_dir, args.depth_format, args.depth_max)

    object_name = args.object_name
    object_position_source = None
    if args.object_position is not None and args.input_object:
        obj_pos, placed_root = place_imported_object(
            scene, args.input_object, args.object_position,
            rotation_z_deg=args.rotation_z_deg, scale=args.scale,
        )
        object_position_source = "import_at_position"
    elif args.object_position is not None:
        obj_pos = Vector(args.object_position)
        object_position_source = "object_position"
    elif args.auto_place_object:
        if not args.input_object:
            raise RuntimeError("--input_object is required when --auto_place_object is set.")
        obj_pos = auto_place_object(scene, args.input_object)
        object_position_source = "auto_place_object"
    elif args.object_name:
        obj_pos = object_position_from_name(args.object_name)
        object_position_source = "object_name"
    else:
        raise RuntimeError("Provide one of --object_position, --object_name, or --auto_place_object.")

    # Pre-generate all poses with the same seed so results are identical
    # regardless of how many workers are used.
    all_poses = [
        (idx, sample_pose(obj_pos, args.camera_radius_range, args.camera_direction_offsets, args.hemisphere))
        for idx in range(args.num_images)
    ]
    # Each worker renders only its slice: worker 0 gets idx 0,3,6,...  worker 1 gets 1,4,7,... etc.
    my_poses = [(idx, pose) for idx, pose in all_poses if idx % args.num_workers == args.worker_index]
    print(f"Worker {args.worker_index}/{args.num_workers}: rendering {len(my_poses)}/{args.num_images} images")

    annotations = []
    if args.num_workers > 1:
        ann_path = run_dir / f"annotations_worker{args.worker_index}.json"
    else:
        ann_path = run_dir / "annotations.json"

    for idx, pose in my_poses:
        if depth_output_node is not None:
            scene.frame_set(idx)
            depth_output_node.file_slots[0].path = "img_"
        camera.matrix_world = Matrix.Translation(pose["cam_pos"]) @ pose["rot_matrix"].to_4x4()
        scene.render.filepath = str(images_dir / f"img_{idx:04d}.png")
        bpy.ops.render.render(write_still=True)
        entry = {
            "image": f"images/img_{idx:04d}.png",
            "camera_position": vec(pose["cam_pos"]),
            "radius": pose["radius"],
            "object_position": vec(obj_pos),
            "base_forward": vec(pose["base_forward"]),
            "base_up": vec(pose["base_up"]),
            "offsets_deg": pose["offsets"],
            "final_forward": vec(pose["final_forward"]),
            "final_up": vec(pose["final_up"]),
        }
        if args.render_depth:
            depth_ext = "exr" if args.depth_format == "OPEN_EXR" else "png"
            entry["depth"] = f"depth/img_{idx:04d}.{depth_ext}"
        annotations.append(entry)

        # Incremental save — prevents annotation loss on crash
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

    run_info = {
        "input_scene": str(blend_path),
        "output_dir": str(run_dir),
        "scene_stem": scene_stem,
        "object_name": object_name,
        "run_name": args.run_name,
        "created_at": timestamp,
        "options": {
            "sky_strength": args.sky_strength,
            "object_position": vec(obj_pos),
            "object_position_source": object_position_source,
            "hemisphere": args.hemisphere,
            "camera_radius_range": args.camera_radius_range,
            "camera_direction_offsets": args.camera_direction_offsets,
            "num_images": args.num_images,
            "seed": args.seed,
            "focal_length": args.focal_length,
            "sensor_width": args.sensor_width,
            "sensor_height": args.sensor_height,
            "resolution": args.resolution,
            "render_device": device_info["device"],
            "render_devices": device_info["devices"],
            "gpu_backend": args.gpu_backend,
            "gpu_devices": args.gpu_devices,
            "samples": args.samples,
            "adaptive_sampling": args.adaptive_sampling,
            "adaptive_threshold": args.adaptive_threshold,
            "max_bounces": args.max_bounces,
            "diffuse_bounces": args.diffuse_bounces,
            "glossy_bounces": args.glossy_bounces,
            "transmission_bounces": args.transmission_bounces,
            "volume_bounces": args.volume_bounces,
            "transparent_max_bounces": args.transparent_max_bounces,
            "persistent_data": args.persistent_data,
            "blender_threads": args.blender_threads,
            "render_depth": args.render_depth,
            "depth_format": args.depth_format,
            "depth_max": args.depth_max,
        },
    }

    # Annotations already saved incrementally above; just save run_info
    if args.num_workers > 1:
        if args.worker_index == 0:
            with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)
        print(f"Worker {args.worker_index}: saved {len(annotations)} images to {ann_path}")
    else:
        with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        print(f"Saved {len(annotations)} images to {run_dir}")


if __name__ == "__main__":
    import sys

    raw_argv = getattr(bpy.app, "argv", None)
    if raw_argv is None:
        raw_argv = sys.argv
    raw_argv = list(raw_argv)
    if "--" not in raw_argv and raw_argv and raw_argv[0].endswith(".py"):
        raw_argv = raw_argv[1:]
    main(raw_argv)

"""Example Usage:
CUDA_VISIBLE_DEVICES=5 blender -b -P render_object.py -- --input_scene assets/Koky_LuxuryHouse_0.blend --camera_direction_offsets 0 0 0 --camera_radius_range 8 8 --run_name deterministic_radius_8_pos_fixed --hemisphere --object_position -0.78 5.58 0.56
"""


"""
object_name 기준: blender -b -P render_object.py -- --input_scene assets/scenes/DogWalk.blend --object_name GEO-snowman-head.001 --camera_radius_range 4 6 --num_images 60
object_position 기준: CUDA_VISIBLE_DEVICES=2 blender/blender -b -P render_object.py -- --input_scene assets/scenes/DogWalk.blend --object_position -0.011 0.0364 0.8 --camera_radius_range 2 8 --hemisphere --camera_direction_offsets 15 15 0 --num_images 5000 --render_depth
auto_place 기준: blender -b -P render_object.py -- --input_scene Koky_LuxuryHouse_0.blend --auto_place_object --input_object assets/objects/snow-man_f667c23f-d220-4faf-91bf-7bd339148bfb --camera_radius_range 6 10 --num_images 80

# Single Blender process, use all OPTIX GPUs, faster settings
blender/blender -b -P render_object.py -- \
  --input_scene /home/nas5/jungwooahn/datasets/DronePhotos/assets/scenes/DogWalk.blend \
  --object_position -0.011 0.0364 0.8 \
  --num_images 10000 \
  --gpu_backend OPTIX \
  --camera_radius_range 2 8 \
  --hemisphere \
  --camera_direction_offsets 15 15 0 \
  --samples 32 \
  --adaptive_sampling --adaptive_threshold 0.02 \
  --max_bounces 2 --diffuse_bounces 1 --glossy_bounces 1 --transmission_bounces 1 \
  --persistent_data
  --gpu_devices 6 7
"""
